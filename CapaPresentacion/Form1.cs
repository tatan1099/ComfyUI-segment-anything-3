using CapaNegocios;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Runtime.CompilerServices.RuntimeHelpers;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace CapaPresentacion
{
    public partial class Form1 : Form
    {
        CN_Vendedores oCN_Vendedores = new CN_Vendedores();
        private bool Editar = false;
        validaciones validaciones = new validaciones();
        DataTable Tabla = new DataTable();
        public Form1()
        {
            InitializeComponent();

        }
        public class cVendedor
        {
            public static int CodVendedor;
        }


        private void BtnVendedores_Click(object sender, EventArgs e)
        {

        }

        private void BtnGuardar_Click(object sender, EventArgs e)
        {

            SqlConnection conexion = new SqlConnection("Server=TATAN\\SQLEXPRESS;Database=NaturVida;Integrated Security=true");

            Form Vende = new InicioVendedores();
            conexion.Open();

            SqlCommand comando = new SqlCommand("SELECT Usuario, Contraseña FROM Vendedores WHERE Usuario =@Usuario AND Contraseña=@Contraseña", conexion);
            if (TxtUsuario.Text.Trim().ToLower() == "" | TxtUsuario.Text.Trim().ToLower() == "")
            {
                MessageBox.Show("Error ingrese los datos");
            }
            else
            {
                comando.Parameters.AddWithValue("@Usuario", TxtUsuario.Text);
                comando.Parameters.AddWithValue("@Contraseña", validaciones.Encriptacion(TxtContrasena.Text));

            }

            SqlDataReader lector = comando.ExecuteReader();
            if (lector.Read())
            {
                conexion.Close();
                this.Hide();
                Vende.Show();
                Global.Id = 5;
                MessageBox.Show(Global.Id.ToString());
            }


            else
            {
                MessageBox.Show("Datos incorrectos", "ERROR", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void TxtContrasena_KeyDown(object sender, KeyEventArgs e)
        {
           

        }
    }
}
   





