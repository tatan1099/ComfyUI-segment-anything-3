using CapaEntidades;
using CapaNegocios;
using iTextSharp.text.pdf;
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
    public partial class Factura : Form
    {
        CN_Clientes oCN_Clientes = new CN_Clientes();
        CN_Productos oCN_Prodcutos = new CN_Productos();
        CN_Factura oCN_Factura = new CN_Factura();
        CE_Productos Productoos = new CE_Productos();
        CN_DetalleFactura oCN_DetalleFactura = new CN_DetalleFactura(); 
        DataTable tabla = new DataTable();
        int factura = 0;
        List<int> Lista = new List<int>();

        public Factura()
        {
            InitializeComponent();
        }

        private void Factura_Load(object sender, EventArgs e)
        {
            buscarCliente();
            buscarProduc();
            MostrarDataGridView();
            MostrarNFacturaa();
        }
        #region Mis METODOS
        private void MostrarNFacturaa()
        {
            if (oCN_Factura.MostrarNFacturas() != " ")
            {
                factura = Convert.ToInt32(oCN_Factura.MostrarNFacturas()) + 1;
                TxtNFactura.Text = factura.ToString();
            }
            else
            {
                TxtNFactura.Text = 1.ToString();
            }


        }
        private void MostrarDataGridView()
        {
            tabla.Columns.Add("Codigo Producto");
            tabla.Columns.Add("Producto");
            tabla.Columns.Add("Valor Unitario");
            tabla.Columns.Add("Cantidad");
            tabla.Columns.Add("Subtotal");

            DgvFact.DataSource = tabla;
          //  DgvFact.Columns["Codigo Producto"].Visible= false;  

        }
        private void EliminarDataGridView()
        {
            DataTable dt = (DataTable)DgvFact.DataSource;   
            dt.Clear(); 
        }
       


        private void buscarCliente()
        {
            cmbCliente.DataSource = oCN_Clientes.MostrarClientes();
            cmbCliente.ValueMember = "Documento";
            cmbCliente.DisplayMember = "Nombre";
        }
        private void buscarProduc()
        {
            CmbProducto.DataSource = oCN_Prodcutos.MostrarProductos();
            CmbProducto.ValueMember = "Codigo";
            CmbProducto.DisplayMember = "Descripción";
        }

        private void LimpiarAgregar()
        {
            CmbProducto.SelectedIndex = -1;
            TxtCantidad.Clear();
        }
        private int TotalFactura()
        {
            int Valor_Total = 0;
            foreach (DataRow filas in tabla.Rows)
            {
                Valor_Total += Convert.ToInt32(filas["subtotal"]);
            }
            int Valor = Valor_Total;
            return Valor;   
        }

        private void LimpiarTodoFactura()
        {
            cmbCliente.SelectedIndex = -1;
            CmbProducto.SelectedIndex = -1;
            TxtCantidad.Clear();    
            TxtTotal.Clear();   
        }

        private void BtnAgrgarP_Click(object sender, EventArgs e)
        {
            string producto = CmbProducto.Text.ToString();
            string producto1 = CmbProducto.SelectedValue.ToString();
            int Valor_Unitario = 0;
            if (cmbCliente.SelectedIndex >= 0 && CmbProducto.SelectedIndex >= 0 && TxtCantidad.Text.Trim().Length > 0)
            {
                Productoos.Codigo = Convert.ToInt32(CmbProducto.SelectedValue);
                Valor_Unitario = Convert.ToInt32(oCN_Prodcutos.MostrarValorProd(Productoos));
                int CantidadProducto = Convert.ToInt32(oCN_Prodcutos.MostrarCantidadProd(Productoos));                
                int CantidadFactura = Convert.ToInt32(TxtCantidad.Text);
                if (CantidadFactura <= CantidadProducto)
                {
                     
                    cmbCliente.Enabled = false;
                
                    DataRow filas = tabla.NewRow();

                    filas["Codigo Producto"] = producto1;
                    filas["Producto"] = producto;
                    Lista.Add(Convert.ToInt32(CmbProducto.SelectedValue));
                    filas["Cantidad"] = TxtCantidad.Text;                 
                    filas["Valor Unitario"] = Valor_Unitario;
                    filas["Subtotal"] = Convert.ToInt32(TxtCantidad.Text) * Convert.ToInt32(oCN_Prodcutos.MostrarValorProd(Productoos).ToString());
                    tabla.Rows.Add(filas);
                    TxtTotal.Text = TotalFactura().ToString();
                    buscarProduc();
                }
                
                else
                {
                    MessageBox.Show("LA CANTIDAD EXCEDE");
                }
                LimpiarAgregar();
            }
        }


            

        #endregion

        private void BtnTerminarFa_Click(object sender, EventArgs e)
        {
            CE_DetalleFactura DetalleFactura = new CE_DetalleFactura(); 
            CE_Factura factura = new CE_Factura();

            factura.Fecha = DtpFf.Value;
            factura.Documento_Cliente = Convert.ToInt32(cmbCliente.SelectedValue);
            factura.Codigo_Vendedor = 1542;
            oCN_Factura.InsertarFacturas(factura);
            foreach (DataRow filas in tabla.Rows)
            {
                
                DetalleFactura.Codigo_Productos = Convert.ToInt32(filas["Codigo Producto"]);
                DetalleFactura.Cantidad = Convert.ToInt32(filas["Cantidad"]);
                oCN_Factura.Descontarcantidas(DetalleFactura);
                
                DetalleFactura.Codigo_Productos = Convert.ToInt32(filas["Codigo Producto"]);
                DetalleFactura.Id = Convert.ToInt32(TxtNFactura.Text);
                DetalleFactura.Cantidad = Convert.ToInt32(filas["Cantidad"]);
                DetalleFactura.Valor_Unidad = Convert.ToInt32(filas["Valor Unitario"]);
                oCN_DetalleFactura.InsertarDetalleFactura(DetalleFactura);
            
            }
            cmbCliente.Enabled = true;
            MessageBox.Show("se agrego factura a base de datos");

            MostrarNFacturaa();
            EliminarDataGridView();
            LimpiarTodoFactura();
           

        }

        private void BtnVolver_Click(object sender, EventArgs e)
        {
            Form pricipal = new InicioVendedores();
            this.Hide();
            pricipal.Show();
        }
        private void Solonum(object objeto)
        {

            System.Windows.Forms.TextBox texto = objeto as System.Windows.Forms.TextBox;
            bool error = false;

            foreach (char c in texto.Text)
            {

                if (!char.IsDigit(c))
                {

                    error = true;
                    break;
                }
            }
            if (error == true)
            {
                errorProvider1.SetError(texto, "Solo se puede numeros");
                BtnAgrgarP.Enabled = false;
            }
            else
            {
                errorProvider1.SetError(texto, "");
                BtnAgrgarP.Enabled = true;
            }

        }

        private void TxtCantidad_TextChanged(object sender, EventArgs e)
        {
            Solonum(sender);
        }
    }
}
