using CapaDatos;
using CapaEntidades;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CapaNegocios
{
    public class CN_DetalleFactura
    {
        CE_DetalleFactura DetalleFactura = new CE_DetalleFactura(); 
        CD_DetalleFactura oCD_DetalleFactura = new CD_DetalleFactura();

        public void InsertarDetalleFactura(CE_DetalleFactura DetalleFactura)
        {
            oCD_DetalleFactura.InsertarFacturaDeta(DetalleFactura);
        }

    }
}
