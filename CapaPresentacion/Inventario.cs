using CapaEntidades;
using CapaNegocios;
using Org.BouncyCastle.Asn1.Utilities;
using Org.BouncyCastle.Crypto.Macs;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CapaPresentacion
{
    public partial class Inventario : Form
    {
        CN_Vendedores oCN_Vendedores = new CN_Vendedores();
        CN_Productos oCN_Productos = new CN_Productos();
        CE_Productos Productoos = new CE_Productos();
        public Inventario()
        {
            InitializeComponent();
        }

        private void Inventario_Load(object sender, EventArgs e)
        {
            
            DgvInventario.DataSource = oCN_Productos.MostrarInventarioMostrarTodo();
           
          
            buscar();
           
        }
        private void buscar()
        {
            comboBox1.DataSource = oCN_Productos.MostrarProductos();
            comboBox1.ValueMember = "Codigo";
            comboBox1.DisplayMember = "Descripción";
        }
        

        private void BtnMostrarTodo_Click(object sender, EventArgs e)
        {
            oCN_Productos.MostrarInventarioMostrarTodo();

        }

        private void BtnMostrarConsulta_Click(object sender, EventArgs e)
        {
            try
            {
                string Codigo;


                Codigo = (comboBox1.SelectedValue.ToString());

                DgvInventario.DataSource = oCN_Productos.MostrarInventarioMostrarTodo();
            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form pricipal = new InicioVendedores();
            this.Hide();
            pricipal.Show();
        }
    }
}
