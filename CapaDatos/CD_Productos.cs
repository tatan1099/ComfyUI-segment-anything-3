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
    public class CD_Productos
    {
        CE_Productos Productos = new CE_Productos();    
        CD_Conexion conexion = new CD_Conexion();
        SqlDataReader Leer;
        SqlCommand comando = new SqlCommand();
        DataTable tabla = new DataTable();
        public DataTable MostrarProductos()
        {
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "MostrarProductos";
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            tabla.Load(Leer);
            conexion.CerrarConexion();
            return tabla;

        }
        public string MostrarValorProductos(CE_Productos Productos)
        {
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "MostrarCantidad2";
            comando.Parameters.AddWithValue("@cod", Productos.Codigo);
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            if (Leer.Read())
            {
                string factura = Leer["Valor_Unidad"].ToString();
                Leer.Close();   
                return factura;             

               
            }
            else
            {
                Leer.Close ();
                return " ";
            }
          
          
        }
        public string MostrarCantidadProductos(CE_Productos Productos)
        {
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "MostrarCantidad1";
            comando.Parameters.AddWithValue("@Cod", Productos.Codigo);
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            if (Leer.Read())
            {
                string factura = Leer["Cantidad"].ToString();
                Leer.Close();
                return factura;


            }
            else
            {
                Leer.Close();
                return " ";
            }


        }

        public DataTable MostrarProductosIl(int Consulta)
        {
            
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "MostrarProductosILe";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Cod", Consulta);
            Leer = comando.ExecuteReader();
            tabla.Load(Leer);
            comando.Parameters.Clear();
            conexion.CerrarConexion();
            return tabla;

        }
        public void InsertarProductos(CE_Productos Productos)
        {

            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_INSERTARPRODUCT";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Codigo",Productos.Codigo);
            comando.Parameters.AddWithValue("@Descri",Productos.Descripción);
            comando.Parameters.AddWithValue("@ValUnd",Productos.Valor_Unidad);
            comando.Parameters.AddWithValue("@Cantida",Productos.Cantidad);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }
        
        public void EditarProductos(CE_Productos Productos)
        {

            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_ACTUALIZARPROD";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Cod", Productos.Codigo);
            comando.Parameters.AddWithValue("@CodNew", Productos.Codigo);
            comando.Parameters.AddWithValue("@Descri", Productos.Descripción);
            comando.Parameters.AddWithValue("@ValUnd", Productos.Valor_Unidad);
            comando.Parameters.AddWithValue("@Cant", Productos.Cantidad);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }
        public void EliminarProductos(CE_Productos Productos)
        {

            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_ELIMINARPROD";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Cod", Productos.Codigo);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }
        public DataTable MostrarInventarioConsultar(string prod)
        {
            DataTable tabla = new DataTable();
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_BUSCARINVENTARIO";
            comando.Parameters.AddWithValue("@Prod",prod );
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            comando.Parameters.Clear();
            tabla.Load(Leer);
            conexion.CerrarConexion();
            return tabla;
        }


        public DataTable MostrarInventarioMostrarTodo()
        {
            DataTable tabla = new DataTable();
            comando.Parameters.Clear();
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_BUSCARINVENTARIOS";
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            comando.Parameters.Clear();
            tabla.Load(Leer);
            conexion.CerrarConexion();
            return tabla;
        }





    }
}
