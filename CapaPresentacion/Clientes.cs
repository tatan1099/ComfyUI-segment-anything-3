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

namespace CapaPresentacion
{
    public partial class Clientes : Form
    {

        public Clientes()
        {
            InitializeComponent();
        }
        CE_Clientes oCE_Clientes = new CE_Clientes();
        CN_Clientes oCN_Clientes = new CN_Clientes();
        validaciones oValidaciones = new validaciones();    
        private void DgvCClientes_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
          
        }

        private void BtnConsultartodo_Click(object sender, EventArgs e)
        {
            oCN_Clientes.MostrarClientes();
        }
        private void CargarComoboConsultar()
        {
            cmbCC.DataSource = oCN_Clientes.MostrarClientes();
            cmbCC.ValueMember = "Documento";
            cmbCC.DisplayMember = "Nombre";
        }
        private void CargarComoboEditar()
        {
            CmbModificarClientes.DataSource = oCN_Clientes.MostrarClientes();
            CmbModificarClientes.ValueMember = "Documento";
            CmbModificarClientes.DisplayMember = "Nombre";
        }
        private void CargarComoboEliminar()
        {
            cmbEliminarClientes.DataSource = oCN_Clientes.MostrarClientes();
            cmbEliminarClientes.ValueMember = "Documento";
            cmbEliminarClientes.DisplayMember = "Nombre";
        }

        private void Clientes_Load(object sender, EventArgs e)
        {
            CargarComoboEliminar(); 
            CargarComoboConsultar();
            CargarComoboEditar();
            DgvCClientes.DataSource = oCN_Clientes.MostrarClientes();
            DgvEditarClientes.DataSource = oCN_Clientes.MostrarClientes();


        }

        private void TpgCopnsultar_Click(object sender, EventArgs e)
        {

        }

        private void BtnConsultar_Click(object sender, EventArgs e)
        {
            try
            {
                int Codigo;


                Codigo = Convert.ToInt32(cmbCC.SelectedValue.ToString());

                cmbCC.DataSource = oCN_Clientes.MostrarClientesii(Codigo);
            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }
        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void BtnInsertar_Click(object sender, EventArgs e)
        {

            if (TxtDocumentoInsertar.Text.Trim().Length > 0 && TxtNombreInsertar.Text.Trim().Length > 0 && TxtDireccionInsertar.Text.Trim().Length > 0 && TxtTelefonoInsertar.Text.Trim().Length > 0 && TxtCorreoInsertar.Text.Trim().Length > 0)
            {

                if (oValidaciones.ValidarCorreo(TxtCorreoInsertar.Text) == true)
                {
                    oCE_Clientes.Documento = Convert.ToInt32(TxtDocumentoInsertar.Text);
                    oCE_Clientes.Nombre = TxtNombreInsertar.Text;
                    oCE_Clientes.Direccion = TxtDireccionInsertar.Text;
                    oCE_Clientes.Telefono = TxtTelefonoInsertar.Text;
                    oCE_Clientes.Correo = TxtDireccionInsertar.Text;
                    oCN_Clientes.InsertarClientesN(oCE_Clientes);
                    MessageBox.Show("ae inserto correctamente");
                    LblValidarCorreo.Text = "Correo Valido";
                    LblValidarCorreo.ForeColor = Color.Green;
                    MessageBox.Show("Ingreso Correctamente");
              
                    LblValidarCorreo.Text = "-";
                    MessageBox.Show("Se ha registrado correctamente el cliente", "GUARDADO", MessageBoxButtons.OK, MessageBoxIcon.Information);


                }
                else if (oValidaciones.ValidarCorreo(TxtCorreoInsertar.Text) == false)
                {
                    LblValidarCorreo.Text = "Correo Invalido";
                    LblValidarCorreo.ForeColor = Color.Red;
                }

            }
            else
            {

                MessageBox.Show("Digite las cosas");

            }
        

        }
        private void BtnEditar_Click(object sender, EventArgs e)
        {
            try

            {
                oCE_Clientes.Documento = Convert.ToInt32(TxtDocumentoEditar.Text);
                oCE_Clientes.Nombre = TxtNombreEditar.Text;
                oCE_Clientes.Direccion = TxtDireccionEditar.Text;
                oCE_Clientes.Telefono = TxtTelefonoEditar.Text;
                oCE_Clientes.Correo = TxtCorreoEditar.Text;
                oCN_Clientes.EditarClientesN(oCE_Clientes);
                MessageBox.Show("Editado Correctamente");
            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }
        }
        private void BtnConsultarEditarCliente_Click(object sender, EventArgs e)
        {
            if (CmbModificarClientes.SelectedIndex >= 0)
            {
                TxtDocumentoEditar.Text = DgvEditarClientes.CurrentRow.Cells["Documento"].Value.ToString();
                TxtNombreEditar.Text = DgvEditarClientes.CurrentRow.Cells["Nombre"].Value.ToString();
                TxtDireccionEditar.Text = DgvEditarClientes.CurrentRow.Cells["Direccion"].Value.ToString();
                TxtTelefonoEditar.Text = DgvEditarClientes.CurrentRow.Cells["Telefono"].Value.ToString();
                TxtCorreoEditar.Text = DgvEditarClientes.CurrentRow.Cells["Correo"].Value.ToString();

            }
        }

        private void BtnEliminar_Click(object sender, EventArgs e)
        {
            try
            {
                if (MessageBox.Show(" Quieres eliminar", "Message", MessageBoxButtons.YesNo) == DialogResult.Yes)
                {
                    int Codigo;
                    Codigo = Convert.ToInt32(cmbEliminarClientes.SelectedValue.ToString());
                    oCN_Clientes.EliminarClientes(oCE_Clientes);

                    MessageBox.Show("Eliminado Correctamente");


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

        //PARA VALIDAR CAJAS DE TEXTO CON LETRAS
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

        private bool ValidacionGuardar(TextBox TxtDocumentoInsertar, TextBox TxtNombreInsertar, TextBox TxtTelefonoInsertar)
        {
            bool docvalido = !Solonum(TxtDocumentoInsertar);
            bool nombrevalido = !LetraEspacio(TxtNombreInsertar);
            bool numvalido = !Solonum(TxtTelefonoInsertar);


            bool validacionExitosa = docvalido && nombrevalido && numvalido;

            BtnInsertar.Enabled = validacionExitosa;

            return validacionExitosa;
        }

        private bool ValidacionEditar(TextBox TxtDocumentoInsertar, TextBox TxtNombreInsertar, TextBox TxtTelefonoInsertar)
        {

            bool descripcionValida = !Solonum(TxtDocumentoInsertar);
            bool valorValido = !LetraEspacio(TxtNombreInsertar);
            bool cantidadValida = !Solonum(TxtTelefonoInsertar);

            bool validacionExitosa = descripcionValida && valorValido && cantidadValida;

            BtnEditar.Enabled = validacionExitosa;

            return validacionExitosa;

        }

        private void TxtDocumentoInsertar_TextChanged(object sender, EventArgs e)
        {
            bool validacionExitosa = ValidacionGuardar(TxtDocumentoInsertar, TxtNombreInsertar, TxtTelefonoInsertar);

            if (validacionExitosa && !string.IsNullOrEmpty(TxtDocumentoInsertar.Text))
            {
                bool exists = oValidaciones.CheckDocExists(Convert.ToInt32(TxtDocumentoInsertar.Text));
                if (exists)
                {
                    LblVerificvarDocumento.Text = "Ya existe un cliente con ese documento";
                }
                else
                {
                    LblVerificvarDocumento.Text = "";
                }
            }
        }

        private void TxtNombreInsertar_TextChanged(object sender, EventArgs e)
        {
            ValidacionGuardar(TxtDocumentoInsertar, TxtNombreInsertar, TxtTelefonoInsertar);
        }

        private void TxtTelefonoInsertar_TextChanged(object sender, EventArgs e)
        {
            ValidacionGuardar(TxtDocumentoInsertar, TxtNombreInsertar, TxtTelefonoInsertar);
        }

        private void TxtTelefonoEditar_TextChanged(object sender, EventArgs e)
        {
            ValidacionEditar(TxtDocumentoEditar, TxtNombreEditar, TxtTelefonoEditar);
        }

        private void TxtDocumentoEditar_TextChanged(object sender, EventArgs e)
        {
           
        }
    }
}
