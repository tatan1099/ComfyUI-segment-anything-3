using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CapaDatos
{
    public class CD_Conexion
    {
        private SqlConnection conexion = new SqlConnection("Server=TATAN\\SQLEXPRESS;Database=NaturVida; Integrated Security=true");

        public SqlConnection AbrirConexion()
        {
            if (conexion.State == ConnectionState.Closed)

                conexion.Open();

            return conexion;
        }
        public SqlConnection CerrarConexion()
        {
            if (conexion.State == ConnectionState.Closed)

                conexion.Close();

            return conexion;
        }
    }
}
