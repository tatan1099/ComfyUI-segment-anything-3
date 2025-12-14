using CapaEntidades;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace CapaDatos
{
    public class CD_Factura
    {
        CD_Conexion conexion = new CD_Conexion();
        SqlDataReader Leer;
        SqlCommand comando = new SqlCommand();
        DataTable tabla = new DataTable();
        public string MostrarNFactura()
        {
            tabla.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "BuscarNFactu";
            comando.CommandType = CommandType.StoredProcedure;
            Leer = comando.ExecuteReader();
            if(Leer.Read() == true)
            {
                string factura = Leer["IdFactu"].ToString();
                Leer.Close();
                return factura;
            }
            else
            {
                Leer.Close();
                return" ";
            }
            

        }
        public void InsertarFactura(CE_Factura Factura)
        {
            comando.Parameters.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_AGGFACT";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Fech",Factura.Fecha);
            comando.Parameters.AddWithValue("@DoClient", Factura.Documento_Cliente);
            comando.Parameters.AddWithValue("@CodVende", Factura.Codigo_Vendedor);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }
        public void InsertarDetalleFactura(CE_DetalleFactura detalleFactura)
        {
            comando.Parameters.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_AGGFACTDETA";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@IDFac", detalleFactura.Id);
            comando.Parameters.AddWithValue("@CodProd", detalleFactura.Codigo_Productos);
            comando.Parameters.AddWithValue("@Cant", detalleFactura.Cantidad);
            comando.Parameters.AddWithValue("@ValUnidad", detalleFactura.Valor_Unidad);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }

        public void DescontarCantidad(CE_DetalleFactura detalleFactura)
        {
            comando.Parameters.Clear();
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_DESCONTARCANT";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@Codprod", detalleFactura.Codigo_Productos);
            comando.Parameters.AddWithValue("@Cantidad", detalleFactura.Cantidad);        
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();

        }





    }
}
