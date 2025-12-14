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

namespace CapaPresentacion
{
    public partial class InicioVendedores : Form
    {
        public InicioVendedores()
        {
            InitializeComponent();
        }

        private void BtnDetallesFactura_Click(object sender, EventArgs e)
        {
            Form detalle = new dF();
            this.Hide();
            detalle.Show();
        }

        private void BtnVendedores_Click(object sender, EventArgs e)
        {

        
            Form ven = new Vendedoresp();
            
            this.Hide();
            ven.Show();


        }

        private void vendedoresToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }

        private void gestionToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form ven = new Vendedoresp();
            ven.TopLevel = false;
            panel1.Controls.Add(ven);

            ven.Show();
        }

        private void productosToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form pro = new Productos();
            pro.TopLevel = false;
            panel1.Controls.Add(pro);
            pro.Show(); 
        }

        private void produuctosToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            Form pro = new Productos();
            pro.TopLevel = false;
            panel1.Controls.Add(pro);
            pro.Show();
        }

        private void facturaToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            Form factura = new Factura();
            factura.TopLevel = false;
            panel1.Controls.Add(factura);          
            factura.Show(); 
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void backeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form Backe = new Backe();   
            this.Hide();
            Backe.Show();
        }

        private void clientesToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form Cliente = new Clientes();
            this.Hide();
            Cliente.Show(); 

        }

        private void inventarioToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form Inve = new Inventario();
            this.Hide();
            Inve.Show();    
        }

        private void clientesToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            Form Cliente = new Clientes();
            Cliente.TopLevel = false;
            panel1.Controls.Add(Cliente);     
            Cliente.Show();
        }

        private void facturaToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            Form factura = new Factura();
            factura.TopLevel = false;
            panel1.Controls.Add(factura);
            factura.Show();
        }

        private void inventarioToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            this.panel1.Controls.Clear();
            Form Inve = new Inventario();
            Inve.TopLevel = false;
            panel1.Controls.Add(Inve);          
            Inve.Show();
        }

        private void vendedoresToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            this.panel1.Controls.Clear();
            Form Vende = new Vendedoresp();
            Vende.TopLevel = false;
            panel1.Controls.Add(Vende);
            Vende.Show();
        }

        private void facturaToolStripMenuItem1_Click_1(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            this.panel1.Controls.Clear();
            Form factu = new Factura();
            factu.TopLevel = false;
            panel1.Controls.Add(factu);
            factu.Show();
        }

        private void inventarioToolStripMenuItem1_Click_1(object sender, EventArgs e)
        {
            this.panel1.Controls.Clear();
            this.panel1.Controls.Clear();
            Form Inve = new Inventario();
            Inve.TopLevel = false;
            panel1.Controls.Add(Inve);
            Inve.Show();
        }
    }
}
