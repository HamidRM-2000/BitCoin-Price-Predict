using Bitcoin_Price_Prediction.Model;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bitcoin_Price_Prediction.ConsoleApp
{
   public static class ModelBuilder
   {
      private static string DATA_PATH = "DataSet\\BTC-USD.csv";
      private static string MODEL_PATH = "Model.zip";
      public static void CreateModel()
      {
         MLContext context = new MLContext(0);
         //Load Data
         IEnumerable<ModelInput> data = ConvertData(context);

         IDataView trainData = context.Data.LoadFromEnumerable(data.Where(x => x.Date.Year == 2019));
         IDataView testData = context.Data.LoadFromEnumerable(data.Where(x => x.Date.Year == 2020));
         //Build PipeLine
         IEstimator<ITransformer> pipeLine = BuildPipeLine(context);
         //Train Model
         ITransformer model = TrainModel(context, pipeLine, trainData);
         //Evaluate Model
         Evaluate(context,model, testData);

         ITransformer newModel = TrainModel(context, pipeLine, testData);
         //Save the Model
         SaveModel(context, newModel);
      }

      private static void Evaluate(MLContext context,ITransformer model, IDataView testData)
      {
         Console.WriteLine("------------------------------- Evaluating Model metrics -------------------------------");
         var predictions = model.Transform(testData);

         IEnumerable<float> actualPrices =
         context.Data.CreateEnumerable<ModelInput>(testData, true)
        .Select(observed => observed.WeightedPrice);

         IEnumerable<float> forecastPrices =
         context.Data.CreateEnumerable<ModelOutput>(predictions, true)
        .Select(prediction => prediction.ForecastedPrice[0]);

         var metrics = actualPrices.Zip(forecastPrices, (actualValue, forecastValue) => actualValue - forecastValue);

         var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
         var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

         Console.WriteLine($"Mean Absolute Error : {MAE:F3}");
         Console.WriteLine($"Root Mean Squared Error : {RMSE:F3}");

      }
      private static void SaveModel(MLContext context, ITransformer model)
      {
         TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecastEngine = model.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);
         forecastEngine.CheckPoint(context, MODEL_PATH);
      }

      private static ITransformer TrainModel(MLContext context, IEstimator<ITransformer> pipeLine, IDataView dataView)
      {
         Console.WriteLine("------------------------------- Start training -------------------------------");
         var model = pipeLine.Fit(dataView);
         Console.WriteLine("------------------------------- training Finished -------------------------------");
         return model;
      }

      private static IEstimator<ITransformer> BuildPipeLine(MLContext context)
      {
         var trainer = context.Forecasting.ForecastBySsa(
             outputColumnName: "ForecastedPrice",
             inputColumnName: "Label",
             windowSize: 2,
             seriesLength: 30,
             trainSize: 365,
             horizon: 7,
             confidenceLevel: 0.95f,
             confidenceLowerBoundColumn: "LowerBoundPrice",
             confidenceUpperBoundColumn: "UpperBoundPrice"
             );
         return trainer;
      }

      private static IEnumerable<ModelInput> ConvertData(MLContext context)
      {
         List<ModelInput> result = new List<ModelInput>();
         var rawData = File.ReadAllLines(DATA_PATH).Skip(1);
         foreach (var row in rawData)
         {
            var splitedRow = row.Split(',');

            float.TryParse(splitedRow[5].ToString(), out float price);

            result.Add(new ModelInput()
            {
               Date = Convert.ToDateTime(splitedRow[0]),
               WeightedPrice = price
            });
         }
         return result.Where(x => x.Date.Year >= 2018);
      }
   }
}
