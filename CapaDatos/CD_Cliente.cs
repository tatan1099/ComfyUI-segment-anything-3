using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CapaEntidades;

namespace CapaDatos
{
    public class CD_Cliente
    {
        CE_Clientes Clientes = new CE_Clientes();   

        CD_Conexion conexion = new CD_Conexion();
        SqlDataReader Leer;
        SqlCommand comando = new SqlCommand();
        DataTable tabla = new DataTable();
        public DataTable MostrarClientes()
        {
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "MostrarClientes";
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            tabla.Load(Leer);
            conexion.CerrarConexion();
            return tabla;

        }
        public DataTable MostrarClientesIl(int Consulta)
        {

            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "MostrarClientesILe";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Docu", Consulta);
            Leer = comando.ExecuteReader();
            tabla.Load(Leer);
            comando.Parameters.Clear();
            conexion.CerrarConexion();
            return tabla;

        }
        public void InsertarClientes(CE_Clientes Clientes)
        {

            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_INSERTARCLIENT";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Docu", Clientes.Documento);
            comando.Parameters.AddWithValue("@Nombre", Clientes.Nombre);
            comando.Parameters.AddWithValue("@Direccion", Clientes.Direccion);
            comando.Parameters.AddWithValue("@Tel", Clientes.Telefono);
            comando.Parameters.AddWithValue("@Corr", Clientes.Correo);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }
        public void EditarClientes(CE_Clientes Clientes)
        {
            comando.Parameters.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_ACTUALCLIENT";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Docu", Clientes.Documento);
            comando.Parameters.AddWithValue("@DocuNew", Clientes.Documento);
            comando.Parameters.AddWithValue("@Nombre", Clientes.Nombre);
            comando.Parameters.AddWithValue("@Direccion", Clientes.Direccion);
            comando.Parameters.AddWithValue("@Tel", Clientes.Telefono);
            comando.Parameters.AddWithValue("@Corr", Clientes.Correo);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }
        public void EliminarClientes(CE_Clientes Clientes)
        {
            comando.Parameters.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_ELIMINARCLIENT";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Docu", Clientes.Documento);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }





    }
}
