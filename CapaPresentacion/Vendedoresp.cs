using CapaEntidades;
using CapaNegocios;
using Org.BouncyCastle.Asn1.Utilities;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using TextBox = System.Windows.Forms.TextBox;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace CapaPresentacion
{
    public partial class Vendedoresp : Form
    {
        
        CN_Vendedores oCN_Vendedores = new CN_Vendedores();
        CE_Vendedores Vendedores = new CE_Vendedores();
        validaciones oValidaciones = new validaciones();    
        public Vendedoresp()
        {
            InitializeComponent();
        }

        
       
        private void Mostrareditarvende()
        {
            CE_Vendedores Vendedores = new CE_Vendedores();
            Vendedores.Codigo = Convert.ToInt32(CMBModi.SelectedValue);
            DgvMostrar.DataSource = oCN_Vendedores.MostrarTodaTablaEditarVende(Vendedores);
            DgvMostrar.Visible = false; 
        }

       
        private void Vendedoresp_Load(object sender, EventArgs e)
        {
            DgvMostrar.DataSource = oCN_Vendedores.MostrarVende();
            DgvInser.DataSource = oCN_Vendedores.MostrarVende();
           
            Dgvmo.DataSource = oCN_Vendedores.MostrarVende();
            buscarConsulta();       
            buscarvAv();
            cargarvomoboactua();
        }

        private void BtnGuardar_Click(object sender, EventArgs e)
        {

            if (TxtCodigo.Text.Trim().Length > 0 && TxtUsuario.Text.Trim().Length > 0 && TxtContrasena.Text.Trim().Length > 0 && TxtContrasena.Text.Trim().Length > 0)
            {
                MessageBox.Show("Ya existe un codigo igual");
                try
                {
                    bool exists = oValidaciones.CheckCodeVender(Convert.ToInt32(TxtCodigo.Text));
                    bool existedscrip = oValidaciones.CheckUserExists(TxtUsuario.Text);
                    if (exists)
                    {
                        LblVerificarCodigo.Text = "Ya existe un producto con ese codigo";
                    }
                    else if (existedscrip)
                    {
                        LblVerificarCodigo.Text = "Ya existe un producto con esa descripción";
                    }
                    else
                    {
                        Vendedores.Codigo = Convert.ToInt32(TxtCodigo.Text);
                        Vendedores.Usuario = TxtUsuario.Text;
                        Vendedores.Contraseña = TxtContrasena.Text;
                        Vendedores.Nombre = TxtNombre.Text;
                        oCN_Vendedores.InsertarVendedores(Vendedores);
                        limpiar();
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
            else
            {
                MessageBox.Show("Digite las cosas");
            }
        }


        private void BtnEditar_Click(object sender, EventArgs e)
        {
            Mostrareditarvende();
            Vendedores.Codigo = Convert.ToInt32(TxtCodigoe.Text);
            Vendedores.Usuario = TxtUsuarioE.Text;
            Vendedores.Contraseña = TxtContrasenaE.Text;
            Vendedores.Nombre = TxtNombreE.Text;

            oCN_Vendedores.EditarVende(Vendedores);
            Mostrareditarvende();
            MessageBox.Show("Se edito Correstctamente");
            limpiar();
        }
     
         private void MostrarvendeI()
         {
            /*
            DgvMostrar.DataSource = oCN_Vendedores.MostrarVendere(comboBox1.Text);
            */
         }

         private void BtnConsultart_Click(object sender, EventArgs e)
         {
                oCN_Vendedores.MostrarVende();
            
         }

        
        //comobo
         private void buscarConsulta()
         {
            CmMostrar.DataSource = oCN_Vendedores.MostrarVende();
            CmMostrar.ValueMember = "Codigo";
            CmMostrar.DisplayMember = "Usuario";
         }

         private void buscarvAv()
         {
                CmbEliminar.DataSource = oCN_Vendedores.MostrarVende();
                CmbEliminar.ValueMember = "Codigo";
                CmbEliminar.DisplayMember = "Usuario";

         }
        private void cargarvomoboactua()
        {
            CMBModi.DataSource = oCN_Vendedores.MostrarVende();
            CMBModi.DisplayMember = "Usuario";
            CMBModi.ValueMember = "Codigo";
            CMBModi.SelectedIndex = -1;

        }

        private void BtnConsultar1_Click(object sender, EventArgs e)
        {
            try
            {
                int Codigo;
                Codigo = Convert.ToInt32(CmMostrar.SelectedValue.ToString());

                Dgvmo.DataSource = oCN_Vendedores.BuscarVende(Codigo);
            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }

        }
      


        private void BtnConsultar_Click(object sender, EventArgs e)
        {
            oCN_Vendedores.MostrarVende();

            

        }
        private void limpiar()
        {
            TxtNombreE.Clear();
            TxtCodigoe.Clear();
            TxtUsuarioE.Clear();
            TxtContrasenaE.Clear();
            TxtNombreE.Clear();
            TxtCodigo.Clear();
            TxtContrasena.Clear();  
            TxtUsuario.Clear();
            TxtNombre.Clear();
            
        }

        private void button2_Click(object sender, EventArgs e)
        {
            
           
            if (CMBModi.SelectedIndex >= 0)
            {
                TxtCodigoe.Text = Dgvmo.CurrentRow.Cells["Codigo"].Value.ToString();
                TxtUsuarioE.Text = Dgvmo.CurrentRow.Cells["Usuario"].Value.ToString();
                TxtContrasenaE.Text = Dgvmo.CurrentRow.Cells["Contraseña"].Value.ToString();
                TxtNombreE.Text = Dgvmo.CurrentRow.Cells["Nombre"].Value.ToString();
                        
                

            }
        }
        private void BtnEliminar_Click(object sender, EventArgs e)
        {
            try
            {
                if (MessageBox.Show(" Quieres eliminar", "Message", MessageBoxButtons.YesNo) == DialogResult.Yes)
                {
                    int Codigo;
                    Codigo = Convert.ToInt32(CmbEliminar.SelectedValue.ToString());
                    oCN_Vendedores.EliminarVende(Vendedores);

                    MessageBox.Show("Eliminado Correctamente");
                    limpiar();

                }

                else
                {
                    MessageBox.Show("No se elimino");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }


           
        }

        private void TxtContrasenaE_TextChanged(object sender, EventArgs e)
        {
            TxtContrasenaE.Enabled = true;
        }

        private void BtnVolver_Click(object sender, EventArgs e)
        {
            Form pricipal = new InicioVendedores();
            this.Hide();
            pricipal.Show();
        }
        private bool Solonum(object objeto)
        {

            TextBox numero = objeto as TextBox;
            bool error = false;

            foreach (char c in numero.Text)
            {

                if (!char.IsDigit(c))
                {

                    error = true;
                    break;
                }
            }
            if (error == true)
            {
                errorProvider1.SetError(numero, "Solo se puede numeros");
            }
            else
            {
                errorProvider1.SetError(numero, "");
            }
            return error;

        }
        private bool SoloLetra(object objeto)
        {
            TextBox texto = objeto as TextBox;
            bool error = false;
            foreach (char c in texto.Text)
            {
                if (!char.IsLetter(c))
                {
                    error = true;
                    break;
                }
            }
            if (error == true)
            {
                errorProvider1.SetError(texto, "Sole se pueden letras");
            }
            else
            {
                errorProvider1.SetError(texto, "");
            }
            return error;
        }
        private bool LetraEspacio(object objeto)
        {
            TextBox texto = objeto as TextBox;
            bool error = false;
            foreach (char c in texto.Text)
            {
                if (!char.IsLetter(c) && c != ' ') // Agrega una condición para permitir espacios en blanco
                {
                    error = true;
                    break;
                }
            }
            if (error == true)
            {
                errorProvider1.SetError(texto, "Solo se pueden letras y espacios en blanco"); // Modifica el mensaje de error para incluir espacios en blanco
            }
            else
            {
                errorProvider1.SetError(texto, "");
            }
            return error;
        }
        private bool LetraNumero(object objeto)
        {
            TextBox texto = objeto as TextBox;
            bool error = false;
            foreach (char c in texto.Text)
            {
                if (!char.IsLetter(c) && !char.IsDigit(c)) // Agrega una condición para permitir espacios en blanco
                {
                    error = true;
                    break;
                }
            }
            if (error == true)
            {
                errorProvider1.SetError(texto, "Solo se pueden letras y numeros"); // Modifica el mensaje de error para incluir espacios en blanco
            }
            else
            {
                errorProvider1.SetError(texto, "");
            }
            return error;
        }
        private bool ValidacionGuardar(TextBox TxtCodigo, TextBox TxtUsuario, TextBox TxtTNombre)
        {
            bool docvalido = !Solonum(TxtCodigo);
            bool usuariovalido = !LetraNumero(TxtUsuario);
            bool nombrevalido = !LetraEspacio(TxtTNombre);



            bool validacionExitosa = docvalido && usuariovalido && nombrevalido;

            BtnGuardar.Enabled = validacionExitosa;

            return validacionExitosa;
        }
        private bool ValidacionEditar(TextBox TxtCodigo, TextBox TxtUsuario, TextBox TxtTNombre)
        {
            bool docvalido = !Solonum(TxtCodigo);
            bool usuariovalido = !LetraNumero(TxtUsuario);
            bool nombrevalido = !LetraEspacio(TxtTNombre);



            bool validacionExitosa = docvalido && usuariovalido && nombrevalido;

            BtnEditar.Enabled = validacionExitosa;

            return validacionExitosa;
        }

        private void TxtCodigo_TextChanged(object sender, EventArgs e)
        {
            bool validacionExitosa = ValidacionGuardar(TxtCodigo, TxtUsuario, TxtNombre);

            if (validacionExitosa && !string.IsNullOrEmpty(TxtCodigo.Text))
            {
                bool exists = oValidaciones.CheckCodeVender(Convert.ToInt32(TxtCodigo.Text));
                if (exists)
                {
                    LblVerificarCodigo.Text = "Ya existe un cliente con ese documento";
                }
                else
                {
                    LblVerificarCodigo.Text = "";
                }
            }
        }

        private void TxtNombre_TextChanged(object sender, EventArgs e)
        {
            ValidacionGuardar(TxtCodigo, TxtUsuario, TxtNombre);
        }

        private void TxtNombreE_TextChanged(object sender, EventArgs e)
        {
            ValidacionEditar(TxtCodigoe, TxtUsuarioE, TxtNombreE);
        }
    }

}
