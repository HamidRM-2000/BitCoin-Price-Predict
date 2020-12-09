using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bitcoin_Price_Prediction.Model
{
   public static class ConsumeModel
   {
      private static string Model_File = "Model.zip";
      private static Lazy<TimeSeriesPredictionEngine<ModelInput, ModelOutput>> PredictionEngine;
      public static ModelOutput Predict(ModelInput modelInput = null, int? horizon = null)
      {
         MLContext context = new MLContext(0);
         PredictionEngine = GetModel(context);
         ModelOutput result = new ModelOutput();
         if (modelInput != null)
         {
            result = PredictionEngine.Value.Predict(modelInput, horizon);
            PredictionEngine.Value.CheckPoint(context, Model_File);
         }
         else
            result = PredictionEngine.Value.Predict(horizon);

         return result;
      }

      private static Lazy<TimeSeriesPredictionEngine<ModelInput, ModelOutput>> GetModel(MLContext context)
      {
         ITransformer model = context.Model.Load(Model_File, out _);
         var engine = model.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);
         return new Lazy<TimeSeriesPredictionEngine<ModelInput, ModelOutput>>(engine);
      }
   }
}
