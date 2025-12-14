using CapaDatos;
using CapaEntidades;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using System.Text;
using System.Threading.Tasks;

namespace CapaNegocios
{
    public class CN_Vendedores
    {
        CD_Vendedores oCD_Vendedores = new CD_Vendedores();
        CE_Vendedores vendedores = new CE_Vendedores();
        public DataTable MostrarVendere(int Consulta) //Se crea un método público de tipo de dato datatable
        {
            DataTable tablaProductos = new DataTable(); //Se crea la variable tabla tipo datatable
            tablaProductos = oCD_Vendedores.BuscarVendedoresi(Consulta); //Se invoca al método mostrar de la instancia oCD_Productos y se guarda en la variable tabla
            return tablaProductos; //Se etorna la tabla
        }
        public DataTable MostrarTodaTablaEditarVende(CE_Vendedores Vendedores)
        {
            DataTable tabla2 = new DataTable();
            tabla2 = oCD_Vendedores.MostrarTodaTablaEsitarVendedores(Vendedores);
            return tabla2;


        }
        public DataTable BuscarVende(int Consulta)
        {
            DataTable tabla = new DataTable();
            tabla = oCD_Vendedores.BuscarVendedoresi(Consulta);
            return tabla;


        }
        public DataTable MostrarVende()
        {
            DataTable tabla = new DataTable();
            tabla = oCD_Vendedores.MostrarVendedores();
            return tabla;
        }
        
        public void InsertarVendedores(CE_Vendedores Vendedores)
        {
            oCD_Vendedores.InsertarVende(Vendedores);
        }
        public void EditarVende(CE_Vendedores Vendedores)
        {
            oCD_Vendedores.EditarVendedores(Vendedores);
        }
        public void EliminarVende(CE_Vendedores Vendedores)
        {
            oCD_Vendedores.EliminarVendedores(Vendedores);
        }
        


        
        
    }
}
