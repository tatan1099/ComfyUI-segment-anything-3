using CapaNegocios;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics.Eventing.Reader;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using CapaEntidades;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;
using System.IO;
using Org.BouncyCastle.Utilities.Collections;
using iTextSharp.text;
using iTextSharp.text.pdf;
using System.Xml.Linq;
using static System.Net.Mime.MediaTypeNames;
using TextBox = System.Windows.Forms.TextBox;

namespace CapaPresentacion
{
    public partial class Productos : Form
    {
        CN_Vendedores oCN_Vendedores = new CN_Vendedores();
        CN_Productos oCN_Productos = new CN_Productos();
       CE_Productos Productoos = new CE_Productos();
        validaciones oValidaciones = new validaciones();    
        
        string CodigoNew = null;
        public Productos()
        {
            InitializeComponent();
        }

        // Productos
        public void MostrarP ()
        {
            DgvMp.DataSource = oCN_Productos.MostrarProductos();
        }
        public void MostrarPi()
        {
            //dataGridView2.DataSource = oCN_Vendedores.MostrarProductosii(TxtConsultar.Text);
        }

        private void buscarvActu()
        {
           CmbMp.DataSource = oCN_Productos.MostrarProductos();
            CmbMp.ValueMember = "Codigo";
            CmbMp.DisplayMember = "Descripción";
        }
        private void buscarvActuualizar()
        {
            CmbEdi.DataSource = oCN_Productos.MostrarProductos();
            CmbEdi.ValueMember = "Codigo";
            CmbEdi.DisplayMember = "Descripción";
        }
        private void buscarvEliminar()
        {
            CmbEli.DataSource = oCN_Productos.MostrarProductos();
            CmbEli.ValueMember = "Codigo";
            CmbEdi.DisplayMember = "Descripción";
        }

        private void button1_Click(object sender, EventArgs e)
        {
            oCN_Productos.InsertarProduct(Productoos);   
        }

        private void dataGridView2_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            MostrarP(); 
        }

     
      

        private void CargarComboBox()
        {

            CmbMp.DataSource = oCN_Productos.MostrarProductos();
            CmbMp.ValueMember = "Codigo";
            CmbMp.DisplayMember = "Descripción";

        }

        private void TpgConsultar_Click(object sender, EventArgs e)
        {

        }

        

        private void BtnConsultar_Click(object sender, EventArgs e)
        {
            try
            {
                int Codigo;
              

                Codigo = Convert.ToInt32(CmbMp.SelectedValue.ToString());  
            
                DgvMp.DataSource = oCN_Productos.MostrarProductosii(Codigo);               
            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }
            
                
      
           
        }


        private void Productos_Load(object sender, EventArgs e)
        {

            DgvMp.DataSource = oCN_Productos.MostrarProductos();

            buscarvActu();
            buscarvActuualizar();
            buscarvEliminar();



        }

        private void button2_Click(object sender, EventArgs e)
        {
            oCN_Productos.MostrarProductos();
        }

        private void TpgEditar_Click(object sender, EventArgs e)
        {

        }

        private void BtnConsultarE_Click(object sender, EventArgs e)
        {
            try
            {
                int Codigo;
               
                
                Codigo = Convert.ToInt32(CmbMp.SelectedValue.ToString());
             
                oCN_Productos.MostrarProductosii(Codigo);

            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }
            if (CmbEdi.SelectedIndex >= 0)
            {
                          
                TxtCodigoE.Text = DgvMp.CurrentRow.Cells["Codigo"].Value.ToString();
                TxtDescripcionE.Text = DgvMp.CurrentRow.Cells["Descripción"].Value.ToString();
                TxtValorUE.Text = DgvMp.CurrentRow.Cells["Valor_Unidad"].Value.ToString();
                TxtCANTIDADE.Text = DgvMp.CurrentRow.Cells["Cantidad"].Value.ToString();

            }
        }

        private void Mostra_Click(object sender, EventArgs e)
        {
            oCN_Productos.MostrarProductos();
        }

        private void BtnEditar_Click(object sender, EventArgs e)
        {

            try 
            {
                Productoos.Codigo = Convert.ToInt32(TxtCodigoE.Text);
                Productoos.Descripción = TxtDescripcionE.Text;
                Productoos.Valor_Unidad = Convert.ToInt32(TxtValorUE.Text); 
                Productoos.Cantidad = Convert.ToInt32(TxtCANTIDADE.Text);
                
                oCN_Productos.EditarProduct(Productoos);
                MessageBox.Show("Editado Correctamente");
            }
            catch (Exception ex)
            {
                MessageBox.Show("error" + ex);
            }
            
        }

        private void BtnEliminar_Click(object sender, EventArgs e)
        {

            try
            {
                if (MessageBox.Show(" Quieres eliminar", "Message", MessageBoxButtons.YesNo) == DialogResult.Yes)
                {
                    int Codigo;
                    Codigo = Convert.ToInt32(CmbEli.SelectedValue.ToString());
                    oCN_Productos.EliminarProduct(Productoos);

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

        private void TxtCodigoE_EnabledChanged(object sender, EventArgs e)
        {
            TxtCodigoE.Enabled = true;
        }
      

        private void CmbEdi_SelectedIndexChanged(object sender, EventArgs e)
        {
          


        }

        private void BtnInsertar_Click(object sender, EventArgs e)
        {
            /*
            Productoos.Codigo= Convert.ToInt32(TxtCodigo.Text);
            Productoos.Descripción = Txt.Text;
            Productoos.Valor_Unidad = Convert.ToInt32(TxtValUnida.Text);
            Productoos.Cantidad= Convert.ToInt32(TxtCantidad.Text);
            oCN_Productos.InsertarProduct(Productoos);
            MessageBox.Show("se inserto datos correctamente");
            */
           
        }

        public void Exportarexcel(DataGridView DgvMP)
        {
            Microsoft.Office.Interop.Excel.Application exportar = new Microsoft.Office.Interop.Excel.Application();
            exportar.Application.Workbooks.Add(true);
            int Indicolumna = 0;
            foreach (DataGridViewColumn colums in DgvMp.Columns) 
            {
                Indicolumna++;
                exportar.Cells[1,Indicolumna] = colums.Name;
            
            }
            int IndiceFila = 0;
            foreach (DataGridViewRow fila in DgvMp.Rows)
            {
                IndiceFila++;
                Indicolumna = 0;
                foreach (DataGridViewColumn Columnas in DgvMp.Columns)
                {
                    Indicolumna++;
                    exportar.Cells[IndiceFila + 1, Indicolumna] = fila.Cells[Columnas.Name].Value;
                }
            }
            exportar.Visible= true;

        }

        private void button3_Click(object sender, EventArgs e)
        {
            Exportarexcel(DgvMp);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            FileStream fileStream = new FileStream(@"C:\Users\Sena CSET\Downloads\expo\pdfProductos.Pdf",FileMode.Create);
            {
                Document document = new Document(PageSize.A4);
                PdfWriter pdfWriter = PdfWriter.GetInstance(document, fileStream);
                document.Open();

                PdfPTable pdfPTable = new PdfPTable(DgvMp.Columns.Count);

                foreach(DataGridViewColumn columns in DgvMp.Columns)
                {
                    PdfPCell cell = new PdfPCell(new Phrase(columns.HeaderText ));
                    pdfPTable.AddCell(cell);

                }
                if(DgvMp.Rows.Count > 0) 
                { 
                    foreach(DataGridViewRow row in DgvMp.Rows) 
                    { 
                        foreach(DataGridViewCell cell in row.Cells) 
                        {
                            pdfPTable.AddCell(cell.Value?.ToString() ?? "");

                        
                        }
                    }
                }
                else
                {

                    MessageBox.Show("se inserto mal ");
                }
                document.Add(pdfPTable);
                document.Close();





            }

        }

        private void DgvMp_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void BtnVolver_Click(object sender, EventArgs e)
        {
            Form pricipal = new InicioVendedores();
            this.Hide();
            pricipal.Show();    

        }

        private void BtnInsertar_Click_1(object sender, EventArgs e)
        {
            if (TxtCodigo.Text.Trim().Length > 0 && TxtDescripcion.Text.Trim().Length > 0 && TxtValorunitario.Text.Trim().Length > 0 && TxtCantidad.Text.Trim().Length > 0)
            {
                try
                {
                    bool exists = oValidaciones.CheckProductCodeExists(Convert.ToInt32(TxtCodigo.Text));
                    bool existedscrip = oValidaciones.CheckProductDescrioExists(TxtDescripcion.Text);
                    if (exists)
                    {
                        LblVerificar.Text = "Ya existe un producto con ese codigo";
                    }
                    else if (existedscrip)
                    {
                        MessageBox.Show("Ya existe un producto con esa descripción");

                    }
                    else
                    {
                        Productoos.Codigo = Convert.ToInt32(TxtCodigo.Text);
                        Productoos.Descripción = TxtDescripcion.Text;
                        Productoos.Valor_Unidad = Convert.ToInt32(TxtValorunitario.Text);
                        Productoos.Cantidad = Convert.ToInt32(TxtCantidad.Text);
                        oCN_Productos.InsertarProduct(Productoos);
                        MessageBox.Show("se inserto datos correctamente");
                    }
                }
                catch(Exception)
                {
                    MessageBox.Show("Digite las cosas");
                }
            }
            else
            {

                MessageBox.Show("Digite las cosas");

            }

        }
        //PARA VALIDAR CAJAS DE TEXTO CON NUMEROS
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
                if (!char.IsLetter(c) && c != ' ' && !char.IsDigit(c)) // Agrega una condición para permitir espacios en blanco
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
        private bool ValidacionGuardar(TextBox TxtCodigo, TextBox TxtDescripcion, TextBox TxtValor, TextBox TxtCantidad)
        {
            bool codigoValido = !Solonum(TxtCodigo);
            bool descripcionValida = !LetraEspacio(TxtDescripcion);
            bool valorValido = !Solonum(TxtValor);
            bool cantidadValida = !Solonum(TxtCantidad);

            bool validacionExitosa = codigoValido && descripcionValida && valorValido && cantidadValida;

            BtnInsertar.Enabled = validacionExitosa;

            return validacionExitosa;
        }



        private void TxtValorunitario_TextChanged(object sender, EventArgs e)
        {
            ValidacionGuardar(TxtCodigo, TxtDescripcion, TxtValorunitario, TxtCantidad);
        }

        private void TxtCodigo_TextChanged(object sender, EventArgs e)
        {
            bool validacionExitosa = ValidacionGuardar(TxtCodigo, TxtDescripcion, TxtValorunitario, TxtCantidad);

            if (validacionExitosa && !string.IsNullOrEmpty(TxtCodigo.Text))
            {
                bool exists = oValidaciones.CheckProductCodeExists(Convert.ToInt32(TxtCodigo.Text));
                if (exists)
                {
                    LblVerificar.Text = "Ya existe un producto con ese codigo";
                }
                else
                {
                    LblVerificar.Text = "";
                }
            }
        }

        private void TxtDescripcion_TextChanged(object sender, EventArgs e)
        {

            bool validacionexitosa = ValidacionGuardar(TxtCodigo, TxtDescripcion, TxtValorunitario, TxtCantidad);

            if (validacionexitosa && !string.IsNullOrEmpty(TxtDescripcion.Text))
            {   
                bool existedscrip = oValidaciones.CheckProductDescrioExists(TxtDescripcion.Text);
                if (existedscrip)
                {
                    LblDescriVerificar.Text = "Ya existe un producto con esa descripcion";
                }
                else
                {
                    LblDescriVerificar.Text = "";
                }
            }

        }

        private void TxtCantidad_TextChanged(object sender, EventArgs e)
        {
            ValidacionGuardar(TxtCodigo, TxtDescripcion, TxtValorunitario, TxtCantidad);
        }
        private bool ValidacionEditar(TextBox TxtDescripcionE, TextBox TxtValorUE, TextBox TxtCANTIDADE)
        {

            bool descripcionValida = !LetraEspacio(TxtDescripcionE);
            bool valorValido = !Solonum(TxtValorUE);
            bool cantidadValida = !Solonum(TxtCANTIDADE);

            bool validacionExitosa = descripcionValida && valorValido && cantidadValida;

            BtnEditar.Enabled = validacionExitosa;

            return validacionExitosa;

        }
        private void TxtCodigoE_TextChanged(object sender, EventArgs e)
        {

        }

        private void TxtDescripcionE_TextChanged(object sender, EventArgs e)
        {
            ValidacionEditar(TxtDescripcionE, TxtValorUE, TxtCANTIDADE);
        }

        private void TxtValorUE_TextChanged(object sender, EventArgs e)
        {
            ValidacionEditar(TxtDescripcionE, TxtValorUE, TxtCANTIDADE);
        }

        private void TxtCANTIDADE_TextChanged(object sender, EventArgs e)
        {
            ValidacionEditar(TxtDescripcionE, TxtValorUE, TxtCANTIDADE);
        }
    }
}
