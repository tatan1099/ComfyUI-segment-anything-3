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
    public class CD_DetalleFactura
    {
        CD_Conexion conexion = new CD_Conexion();
        SqlDataReader Leer;
        SqlCommand comando = new SqlCommand();
        DataTable tabla = new DataTable();

        public void InsertarFacturaDeta(CE_DetalleFactura DetalleFactura)
        {
            comando.Connection = conexion.AbrirConexion();
            comando.CommandText = "SP_AGGFACTDETA";
            comando.CommandType = CommandType.StoredProcedure;
            comando.Parameters.AddWithValue("@IdFac", DetalleFactura.Id);
            comando.Parameters.AddWithValue("@CodProd", DetalleFactura.Codigo_Productos);
            comando.Parameters.AddWithValue("@Cant", DetalleFactura.Cantidad);
            comando.Parameters.AddWithValue("@ValUnidad", DetalleFactura.Valor_Unidad);
            comando.ExecuteNonQuery();
            comando.Parameters.Clear();
            conexion.CerrarConexion();
        }

    }
}
