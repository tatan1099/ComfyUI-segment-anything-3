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
    public partial class Backe : Form
    {
        public Backe()
        {
            InitializeComponent();
        }

        private void abrirToolStripMenuItem_Click(object sender, EventArgs e)
        {
           OpenFileDialog oFD = new OpenFileDialog();  
          //  OPF1.Filter = "Archivo txt (*.txt)|*.txt";
            oFD.Title = "Abrir";
            oFD.InitialDirectory = @"C:\Users\Sena CSET\Downloads\Crud Evaluacion jonathan 222";

            if(oFD.ShowDialog() == DialogResult.OK)
            {
                TxtAbrir.Text = oFD.FileName;
            }
            oFD.Dispose();

        }

        private void Backe_Load(object sender, EventArgs e)
        {

        }

        private void buscarToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FB1.Description = "selecione la ruta";
            FB1.RootFolder = Environment.SpecialFolder.Desktop;
            if(FB1.ShowDialog()==DialogResult.OK)
            {
                TxtBuscar.Text = FB1.SelectedPath;
            
            }
        }

        private void guardarToolStripMenuItem_Click(object sender, EventArgs e)
        {
            
            SFD1.Filter = "Archivo txt (*.txt)|*.txt";
            SFD1.Title = "Guardar";
         
            if (SFD1.ShowDialog() == DialogResult.OK)
            {
                TxtAbrir.Text = SFD1.FileName;
            }
            SFD1.Dispose();
        }
    }
}
