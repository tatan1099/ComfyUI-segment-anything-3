using CapaDatos;
using CapaEntidades;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CapaNegocios
{
    public class CN_Productos
    {
        
        CD_Productos oCD_Productos = new CD_Productos();    
        public DataTable MostrarProductos()
        {
            DataTable tablaProductos = new DataTable();
            tablaProductos = oCD_Productos.MostrarProductos();
            return tablaProductos;
        }
        public DataTable MostrarProductosii(int Consulta)
        {
            DataTable tablaProductos = new DataTable();
            tablaProductos = oCD_Productos.MostrarProductosIl(Consulta);
            return tablaProductos;
        }
        public void InsertarProduct( CE_Productos Productos)
        {
            oCD_Productos.InsertarProductos(Productos);
        }
        public void EditarProduct(CE_Productos Productos)
        {
            oCD_Productos.EditarProductos(Productos);
        }
        public void EliminarProduct(CE_Productos Productos)
        {
            oCD_Productos.EliminarProductos(Productos);
        }
        public string  MostrarValorProd( CE_Productos Productos)
        {

            string tabla = oCD_Productos.MostrarValorProductos(Productos);
            return tabla;   
        }
        public string MostrarCantidadProd(CE_Productos Productos)
        {

            string tabla = oCD_Productos.MostrarCantidadProductos(Productos);
            return tabla;
        }
        public DataTable MostrarInventarioConsultar(string prod)
        {
            DataTable tabla = new DataTable();
            tabla = oCD_Productos.MostrarInventarioConsultar(prod);
            return tabla;
        }

        public DataTable MostrarInventarioMostrarTodo()
        {
            DataTable tabla = new DataTable();
            tabla = oCD_Productos.MostrarInventarioMostrarTodo();
            return tabla;
        }


    }
}
