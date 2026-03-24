import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { BarChart3, Download, Play } from "lucide-react"

interface PredictionResult {
  model: string
  dataset?: string
  rows_predicted: number
  headline: string
  result_ref?: string
}

export function PredictionResultCard({ result, onEvaluate }: {
  result: PredictionResult
  onEvaluate?: (model: string) => void
}) {
  return (
    <Card className="my-2 border-border/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-emerald-500" />
          Prediction Complete
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="text-sm text-muted-foreground">{result.headline}</p>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span>Model: <strong className="text-foreground">{result.model}</strong></span>
          <span>Rows: <strong className="text-foreground">{result.rows_predicted.toLocaleString()}</strong></span>
        </div>
        <div className="flex gap-2 pt-1">
          {result.result_ref && (
            <Button variant="outline" size="sm" className="h-7 text-xs" asChild>
              <a href={`/api/datasets/${result.result_ref}/download`}>
                <Download className="h-3 w-3 mr-1" />
                Download
              </a>
            </Button>
          )}
          {onEvaluate && (
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              onClick={() => onEvaluate(result.model)}
            >
              <Play className="h-3 w-3 mr-1" />
              Evaluate
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
