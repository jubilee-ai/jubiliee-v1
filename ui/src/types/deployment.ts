export interface DeploymentMetrics {
  trainAccuracy?: number
  testAccuracy?: number
  testRocAuc?: number
  valR2?: number
  testR2?: number
  testRmse?: number
  testMae?: number
}

export interface Deployment {
  id: string
  modelName: string
  modelType: string | null
  description: string
  datasetUsed: string[]
  metrics: DeploymentMetrics
  updatedAt: number | null
}
