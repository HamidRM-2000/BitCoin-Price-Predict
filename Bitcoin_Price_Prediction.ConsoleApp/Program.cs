using Bitcoin_Price_Prediction.Model;
using Microsoft.ML;
using System;

namespace Bitcoin_Price_Prediction.ConsoleApp
{
   class Program
   {
      static void Main(string[] args)
      {
         //This line is for creating model
         //ModelBuilder.CreateModel();
         ModelInput input = new ModelInput()
         {
            Date = DateTime.Now,
            WeightedPrice = 18245.52f
         };
         var result = ConsumeModel.Predict(input,horizon: 3);

         Console.WriteLine($"Predicted Bitcoin actual price for is {result.ForecastedPrice[0]:#$}");
         Console.WriteLine($"Predicted Bitcoin lowerbound price for is {result.LowerBoundPrice[0]:#$}");
         Console.WriteLine($"Predicted Bitcoin upperbound price for is {result.UpperBoundPrice[0]:#$}");
      }
   }
}
