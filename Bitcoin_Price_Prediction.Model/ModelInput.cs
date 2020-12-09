using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bitcoin_Price_Prediction.Model
{
   public class ModelInput
   {
      public DateTime Date{ get; set; }


      [ColumnName("Label")]
      public float WeightedPrice { get; set; }
   }
}
