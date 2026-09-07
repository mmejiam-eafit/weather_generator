// Declarative definition of everything actually running in rg-weather-generator,
// reverse-engineered from the resources we created imperatively via `az` this
// session. Written for learning, not (yet) as the live source of truth --
// deploying this from scratch would need globally-unique names picked for the
// two storage accounts (Azure storage account names are a global namespace).
//
// Notable omission: an Azure ML *compute cluster* is NOT declared here. We
// tried one (`az ml compute create`) and it failed: AmlCompute's VM-size
// allow-list tops out at the v5 generation, but every v5 SKU in this
// subscription's eastus2 quota came back `NotAvailableForSubscription` --
// only v7 SKUs (like the E8s_v7 below) are actually provisionable here, and
// AML doesn't support those yet. That's why we fell back to a plain VM
// instead. The AML workspace and datastore below are still real and usable
// for other purposes (e.g. a Compute *Instance*, which is just a managed VM
// and isn't subject to the same allow-list) -- they're just not doing
// anything today.

@description('Region for every resource. eastus2 was chosen after eastus turned out to have dropped the v5 general-purpose SKU family entirely from its catalog.')
param location string = 'eastus2'

@description('SSH public key for the VM admin user.')
param sshPublicKey string

@description('Single IP (CIDR /32) allowed to reach port 22. Azure rejects a push with this open to the world by policy of this template, not just habit -- update this param whenever your IP changes.')
param allowedSshSourceIp string

@description('VM admin username.')
param adminUsername string = 'azureuser'

@description('Name of the data storage account. Globally unique across all of Azure -- pick a fresh one if redeploying.')
param dataStorageAccountName string = 'wxgenarchivee2'

@description('Name of the AML workspace-owned storage account. Globally unique.')
param amlStorageAccountName string = 'mlwweathstoragefa07d228a'

// ---------------------------------------------------------------------------
// Data storage: the training archive lives here, uploaded once via azcopy.
// ---------------------------------------------------------------------------

resource dataStorage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: dataStorageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource dataStorageBlobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: dataStorage
  name: 'default'
}

resource trainingDataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: dataStorageBlobService
  name: 'training-data'
  properties: {
    publicAccess: 'None'
  }
}

// ---------------------------------------------------------------------------
// Azure ML workspace and its dependencies. The `az ml workspace create` CLI
// command auto-provisions all four of these for you in one call; Bicep makes
// you wire them up explicitly, which is honestly the more useful view of
// what "creating a workspace" actually means.
// ---------------------------------------------------------------------------

resource amlKeyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'mlwweathkeyvault16bcc740'
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    accessPolicies: []
    enableSoftDelete: true
  }
}

resource amlAppInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'mlwweathinsightse79f11cc'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: amlLogAnalytics.id
  }
}

// App Insights in "workspace-based" mode requires a Log Analytics workspace
// as its backing store -- this is what actually generates the
// "Application Insights Smart Detection" action group as a side effect; that
// action group isn't declared anywhere, Azure Monitor just creates it.
resource amlLogAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'mlwweathlogalytic1bb29db'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
  }
}

resource amlStorage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: amlStorageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: 'mlw-weather-generator'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'mlw-weather-generator'
    keyVault: amlKeyVault.id
    applicationInsights: amlAppInsights.id
    storageAccount: amlStorage.id
  }
}

// Grants the workspace's own managed identity read access to the *data*
// storage account -- this is what lets the datastore below be
// "credential-less" (no SAS token, no account key checked into anything).
// This is the fix for the exact RBAC gap that broke a user-delegation SAS
// earlier in this project: creating a storage account via the management
// plane does not itself grant blob data-plane access to anyone, including
// a workspace identity -- it has to be assigned explicitly, as here.
resource amlWorkspaceBlobReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(dataStorage.id, amlWorkspace.id, 'Storage Blob Data Reader')
  scope: dataStorage
  properties: {
    principalId: amlWorkspace.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1' // Storage Blob Data Reader
    )
  }
}

resource trainingDataDatastore 'Microsoft.MachineLearningServices/workspaces/datastores@2024-04-01' = {
  parent: amlWorkspace
  name: 'training_data'
  properties: {
    datastoreType: 'AzureBlob'
    accountName: dataStorage.name
    containerName: trainingDataContainer.name
    // No `credentials` block at all == identity-based (credential-less)
    // access, using the role assignment above. This is the Bicep
    // equivalent of the CLI datastore.yml simply omitting `credentials`.
  }
}

// ---------------------------------------------------------------------------
// Networking for the VM: VNet, subnet, NSG (SSH locked to one IP), a static
// public IP, and the NIC that ties them together.
// ---------------------------------------------------------------------------

resource vnet 'Microsoft.Network/virtualNetworks@2023-09-01' = {
  name: 'vm-weather-generatorVNET'
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.0.0.0/16'] }
    subnets: [
      {
        name: 'vm-weather-generatorSubnet'
        properties: { addressPrefix: '10.0.0.0/24' }
      }
    ]
  }
}

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-09-01' = {
  name: 'vm-weather-generatorNSG'
  location: location
  properties: {
    securityRules: [
      {
        name: 'default-allow-ssh'
        properties: {
          priority: 1000
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '22'
          sourceAddressPrefix: allowedSshSourceIp
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

resource publicIp 'Microsoft.Network/publicIPAddresses@2023-09-01' = {
  name: 'vm-weather-generatorPublicIP'
  location: location
  sku: { name: 'Standard' } // Standard SKU = static allocation; this is why the VM's IP hasn't changed across the whole session
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2023-09-01' = {
  name: 'vm-weather-generatorVMNic'
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: { id: vnet.properties.subnets[0].id }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: { id: publicIp.id }
        }
      }
    ]
    networkSecurityGroup: { id: nsg.id }
  }
}

// ---------------------------------------------------------------------------
// The data disk: a separate managed disk, attached to the VM below. Created
// after the fact this session because the OS disk (29GB default) filled up
// mid-download -- 256GB gives real headroom over the ~140GB training archive.
// ---------------------------------------------------------------------------

resource dataDisk 'Microsoft.Compute/disks@2023-10-02' = {
  name: 'disk-weather-generator-data'
  location: location
  sku: { name: 'StandardSSD_LRS' }
  properties: {
    creationData: { createOption: 'Empty' }
    diskSizeGB: 256
  }
}

// ---------------------------------------------------------------------------
// The VM itself. Standard_E8s_v7 was the end of a real search: it's one of
// the few SKUs in this eastus2 subscription with no `NotAvailableForSubscription`
// restriction (checked via `az vm list-skus`) -- most v4/v5/v6 D/E-family
// SKUs came back restricted, and eastus (the other region we checked) has
// dropped the standard x86 v5 general-purpose family from its catalog
// entirely, leaving only ARM and confidential-compute variants there.
// ---------------------------------------------------------------------------

resource vm 'Microsoft.Compute/virtualMachines@2023-09-01' = {
  name: 'vm-weather-generator'
  location: location
  properties: {
    hardwareProfile: {
      vmSize: 'Standard_E8s_v7'
    }
    osProfile: {
      computerName: 'vm-weather-generator'
      adminUsername: adminUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: 'ubuntu-24_04-lts'
        sku: 'server'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
      }
      dataDisks: [
        {
          lun: 0
          createOption: 'Attach'
          managedDisk: { id: dataDisk.id }
        }
      ]
    }
    networkProfile: {
      networkInterfaces: [
        { id: nic.id }
      ]
    }
  }
}
