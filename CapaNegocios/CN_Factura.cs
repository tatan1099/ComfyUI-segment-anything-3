using CapaDatos;
using CapaEntidades;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CapaNegocios
{
    public class CN_Factura
    {
        private CD_Factura oCD_Factura = new CD_Factura();
        CE_Factura Factura = new CE_Factura();  


        public string MostrarNFacturas()
        {
            string factura = oCD_Factura.MostrarNFactura();
            return factura;
        }

        public void InsertarFacturas(CE_Factura Factura)
        {
            oCD_Factura.InsertarFactura(Factura);
        }
        public void Descontarcantidas(CE_DetalleFactura detalleFactura)
        {
            oCD_Factura.DescontarCantidad(detalleFactura);
        }




    }   
}
