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
    public class CN_Clientes
    {
        CD_Cliente oCD_Clientes = new CD_Cliente(); 
        CE_Clientes Clientes = new CE_Clientes();   
        public DataTable MostrarClientes()
        {
            DataTable tablaProductos = new DataTable();
            tablaProductos = oCD_Clientes.MostrarClientes();
            return tablaProductos;
        }
        public DataTable MostrarClientesii(int Consulta)
        {
            DataTable tablaProductos = new DataTable();
            tablaProductos = oCD_Clientes.MostrarClientesIl(Consulta);
            return tablaProductos;
        }
        public void InsertarClientesN(CE_Clientes Clientes)
        {
            oCD_Clientes.InsertarClientes(Clientes);
        }
        public void EditarClientesN(CE_Clientes Clientes)
        {
            oCD_Clientes.EditarClientes(Clientes);
        }
        public void EliminarClientes(CE_Clientes Clientes)
        {
            oCD_Clientes.EliminarClientes(Clientes);
        }

    }
}
