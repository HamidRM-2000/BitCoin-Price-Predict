using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bitcoin_Price_Prediction.Model
{
   public class ModelOutput
   {
      public float[] ForecastedPrice { get; set; }

      public float[] LowerBoundPrice { get; set; }

      public float[] UpperBoundPrice { get; set; }
   }
}
