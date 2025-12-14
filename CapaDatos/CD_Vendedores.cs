using CapaEntidades;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CapaDatos
{
    public class CD_Vendedores
    {
        CD_Conexion conexion= new CD_Conexion();
        SqlDataReader Leer;
        SqlCommand comando = new SqlCommand();
        DataTable tabla = new DataTable();
        CE_Vendedores Vendedores = new CE_Vendedores();

        public DataTable MostrarVendedores()
        {
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "MostrarVendedores";
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            comando.Parameters.Clear();
            tabla.Load(Leer);           
            conexion.CerrarConexion();
            return tabla;          
        }
        public DataTable MostrarTodaTablaEsitarVendedores(CE_Vendedores Vendedores)
        {
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "todatablaVendeEditar";
            comando.Parameters.AddWithValue("@Cod", Vendedores.Codigo);
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            comando.Parameters.Clear();
            tabla.Load(Leer);      
            conexion.CerrarConexion();
            return tabla;

        }

        public DataTable BuscarVendedoresi(int Consulta)
        {
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "BuscarVende8";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Codigo", Consulta);
            Leer = comando.ExecuteReader();
            tabla.Load(Leer);
            comando.Parameters.Clear();
            conexion.CerrarConexion();
            return tabla;

        }
        public void InsertarVende(CE_Vendedores Vendedores)
        {
            comando.Parameters.Clear(); 
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "InsertarVende";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Codigo", Vendedores.Codigo);
            comando.Parameters.AddWithValue("@Usuario", Vendedores.Usuario);
            comando.Parameters.AddWithValue("@Contraseña", Vendedores.Contraseña);
            comando.Parameters.AddWithValue("@Nombre", Vendedores.Nombre);           
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }
        
        public void EditarVendedores(CE_Vendedores Vendedores)
        {
            comando.Parameters.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "EditarVendedores";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Codigo", Vendedores.Codigo);
            comando.Parameters.AddWithValue("@Usuario", Vendedores.Usuario);
            comando.Parameters.AddWithValue("@Contraseña", Vendedores.Contraseña);
            comando.Parameters.AddWithValue("@Nombre", Vendedores.Nombre);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();
        }
        public void EliminarVendedores(CE_Vendedores Vendedores)
        {
            comando.Parameters.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "EliminarVende1";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Codigo", Vendedores.Codigo);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();
        }





    }
}
