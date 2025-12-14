using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CapaEntidades
{
    public class CE_Factura
    {
        public int IdFactu { get; set; }
    
        public DateTime Fecha { get; set; }
        public int Documento_Cliente { get; set; }
        public int Codigo_Vendedor { get; set; }
    }
}
