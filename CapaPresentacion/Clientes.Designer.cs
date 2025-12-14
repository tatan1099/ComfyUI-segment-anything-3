namespace CapaPresentacion
{
    partial class Clientes
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.TpgCopnsultar = new System.Windows.Forms.TabPage();
            this.BtnVolver = new System.Windows.Forms.Button();
            this.cmbCC = new System.Windows.Forms.ComboBox();
            this.BtnConsultar = new System.Windows.Forms.Button();
            this.BtnConsultartodo = new System.Windows.Forms.Button();
            this.DgvCClientes = new System.Windows.Forms.DataGridView();
            this.TpgInsertar = new System.Windows.Forms.TabPage();
            this.LblVerificarNombre = new System.Windows.Forms.Label();
            this.LblVerificvarDocumento = new System.Windows.Forms.Label();
            this.LblValidarCorreo = new System.Windows.Forms.Label();
            this.BtnInsertar = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.TxtCorreoInsertar = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.LblDocumento = new System.Windows.Forms.Label();
            this.TxtTelefonoInsertar = new System.Windows.Forms.TextBox();
            this.TxtDireccionInsertar = new System.Windows.Forms.TextBox();
            this.TxtNombreInsertar = new System.Windows.Forms.TextBox();
            this.TxtDocumentoInsertar = new System.Windows.Forms.TextBox();
            this.TpgModificar = new System.Windows.Forms.TabPage();
            this.BtnEditar = new System.Windows.Forms.Button();
            this.DgvEditarClientes = new System.Windows.Forms.DataGridView();
            this.BtnConsultarEditarCliente = new System.Windows.Forms.Button();
            this.CmbModificarClientes = new System.Windows.Forms.ComboBox();
            this.TxtCorreoEditar = new System.Windows.Forms.TextBox();
            this.TxtTelefonoEditar = new System.Windows.Forms.TextBox();
            this.TxtDireccionEditar = new System.Windows.Forms.TextBox();
            this.TxtNombreEditar = new System.Windows.Forms.TextBox();
            this.TxtDocumentoEditar = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.TpgEliminar = new System.Windows.Forms.TabPage();
            this.BtnEliminar = new System.Windows.Forms.Button();
            this.cmbEliminarClientes = new System.Windows.Forms.ComboBox();
            this.errorProvider1 = new System.Windows.Forms.ErrorProvider(this.components);
            this.errorProvider2 = new System.Windows.Forms.ErrorProvider(this.components);
            this.tabControl1.SuspendLayout();
            this.TpgCopnsultar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.DgvCClientes)).BeginInit();
            this.TpgInsertar.SuspendLayout();
            this.TpgModificar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.DgvEditarClientes)).BeginInit();
            this.TpgEliminar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider2)).BeginInit();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.TpgCopnsultar);
            this.tabControl1.Controls.Add(this.TpgInsertar);
            this.tabControl1.Controls.Add(this.TpgModificar);
            this.tabControl1.Controls.Add(this.TpgEliminar);
            this.tabControl1.Location = new System.Drawing.Point(0, 1);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(897, 445);
            this.tabControl1.TabIndex = 0;
            // 
            // TpgCopnsultar
            // 
            this.TpgCopnsultar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpgCopnsultar.Controls.Add(this.BtnVolver);
            this.TpgCopnsultar.Controls.Add(this.cmbCC);
            this.TpgCopnsultar.Controls.Add(this.BtnConsultar);
            this.TpgCopnsultar.Controls.Add(this.BtnConsultartodo);
            this.TpgCopnsultar.Controls.Add(this.DgvCClientes);
            this.TpgCopnsultar.Location = new System.Drawing.Point(4, 22);
            this.TpgCopnsultar.Name = "TpgCopnsultar";
            this.TpgCopnsultar.Padding = new System.Windows.Forms.Padding(3);
            this.TpgCopnsultar.Size = new System.Drawing.Size(889, 419);
            this.TpgCopnsultar.TabIndex = 0;
            this.TpgCopnsultar.Text = "Consultar";
            this.TpgCopnsultar.Click += new System.EventHandler(this.TpgCopnsultar_Click);
            // 
            // BtnVolver
            // 
            this.BtnVolver.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnVolver.Location = new System.Drawing.Point(616, 16);
            this.BtnVolver.Name = "BtnVolver";
            this.BtnVolver.Size = new System.Drawing.Size(75, 23);
            this.BtnVolver.TabIndex = 4;
            this.BtnVolver.Text = "Volver";
            this.BtnVolver.UseVisualStyleBackColor = false;
            this.BtnVolver.Click += new System.EventHandler(this.BtnVolver_Click);
            // 
            // cmbCC
            // 
            this.cmbCC.FormattingEnabled = true;
            this.cmbCC.Location = new System.Drawing.Point(251, 109);
            this.cmbCC.Name = "cmbCC";
            this.cmbCC.Size = new System.Drawing.Size(121, 21);
            this.cmbCC.TabIndex = 3;
            // 
            // BtnConsultar
            // 
            this.BtnConsultar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultar.Location = new System.Drawing.Point(170, 109);
            this.BtnConsultar.Name = "BtnConsultar";
            this.BtnConsultar.Size = new System.Drawing.Size(75, 23);
            this.BtnConsultar.TabIndex = 2;
            this.BtnConsultar.Text = "Consultar";
            this.BtnConsultar.UseVisualStyleBackColor = false;
            this.BtnConsultar.Click += new System.EventHandler(this.BtnConsultar_Click);
            // 
            // BtnConsultartodo
            // 
            this.BtnConsultartodo.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultartodo.Location = new System.Drawing.Point(404, 109);
            this.BtnConsultartodo.Name = "BtnConsultartodo";
            this.BtnConsultartodo.Size = new System.Drawing.Size(75, 23);
            this.BtnConsultartodo.TabIndex = 1;
            this.BtnConsultartodo.Text = "ConsultarTodo";
            this.BtnConsultartodo.UseVisualStyleBackColor = false;
            this.BtnConsultartodo.Click += new System.EventHandler(this.BtnConsultartodo_Click);
            // 
            // DgvCClientes
            // 
            this.DgvCClientes.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.DgvCClientes.Location = new System.Drawing.Point(178, 163);
            this.DgvCClientes.Name = "DgvCClientes";
            this.DgvCClientes.Size = new System.Drawing.Size(240, 150);
            this.DgvCClientes.TabIndex = 0;
            this.DgvCClientes.CellContentClick += new System.Windows.Forms.DataGridViewCellEventHandler(this.DgvCClientes_CellContentClick);
            // 
            // TpgInsertar
            // 
            this.TpgInsertar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpgInsertar.Controls.Add(this.LblVerificarNombre);
            this.TpgInsertar.Controls.Add(this.LblVerificvarDocumento);
            this.TpgInsertar.Controls.Add(this.LblValidarCorreo);
            this.TpgInsertar.Controls.Add(this.BtnInsertar);
            this.TpgInsertar.Controls.Add(this.label1);
            this.TpgInsertar.Controls.Add(this.TxtCorreoInsertar);
            this.TpgInsertar.Controls.Add(this.label4);
            this.TpgInsertar.Controls.Add(this.label3);
            this.TpgInsertar.Controls.Add(this.label2);
            this.TpgInsertar.Controls.Add(this.LblDocumento);
            this.TpgInsertar.Controls.Add(this.TxtTelefonoInsertar);
            this.TpgInsertar.Controls.Add(this.TxtDireccionInsertar);
            this.TpgInsertar.Controls.Add(this.TxtNombreInsertar);
            this.TpgInsertar.Controls.Add(this.TxtDocumentoInsertar);
            this.TpgInsertar.Location = new System.Drawing.Point(4, 22);
            this.TpgInsertar.Name = "TpgInsertar";
            this.TpgInsertar.Padding = new System.Windows.Forms.Padding(3);
            this.TpgInsertar.Size = new System.Drawing.Size(889, 419);
            this.TpgInsertar.TabIndex = 1;
            this.TpgInsertar.Text = "Insertar";
            // 
            // LblVerificarNombre
            // 
            this.LblVerificarNombre.AutoSize = true;
            this.LblVerificarNombre.Location = new System.Drawing.Point(184, 110);
            this.LblVerificarNombre.Name = "LblVerificarNombre";
            this.LblVerificarNombre.Size = new System.Drawing.Size(0, 13);
            this.LblVerificarNombre.TabIndex = 13;
            // 
            // LblVerificvarDocumento
            // 
            this.LblVerificvarDocumento.AutoSize = true;
            this.LblVerificvarDocumento.Location = new System.Drawing.Point(197, 72);
            this.LblVerificvarDocumento.Name = "LblVerificvarDocumento";
            this.LblVerificvarDocumento.Size = new System.Drawing.Size(0, 13);
            this.LblVerificvarDocumento.TabIndex = 12;
            // 
            // LblValidarCorreo
            // 
            this.LblValidarCorreo.AutoSize = true;
            this.LblValidarCorreo.Location = new System.Drawing.Point(181, 242);
            this.LblValidarCorreo.Name = "LblValidarCorreo";
            this.LblValidarCorreo.Size = new System.Drawing.Size(0, 13);
            this.LblValidarCorreo.TabIndex = 11;
            // 
            // BtnInsertar
            // 
            this.BtnInsertar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnInsertar.Location = new System.Drawing.Point(211, 172);
            this.BtnInsertar.Name = "BtnInsertar";
            this.BtnInsertar.Size = new System.Drawing.Size(75, 23);
            this.BtnInsertar.TabIndex = 10;
            this.BtnInsertar.Text = "Insertar";
            this.BtnInsertar.UseVisualStyleBackColor = false;
            this.BtnInsertar.Click += new System.EventHandler(this.BtnInsertar_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label1.Location = new System.Drawing.Point(7, 242);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(38, 13);
            this.label1.TabIndex = 9;
            this.label1.Text = "Correo";
            // 
            // TxtCorreoInsertar
            // 
            this.TxtCorreoInsertar.Location = new System.Drawing.Point(75, 235);
            this.TxtCorreoInsertar.Name = "TxtCorreoInsertar";
            this.TxtCorreoInsertar.Size = new System.Drawing.Size(100, 20);
            this.TxtCorreoInsertar.TabIndex = 8;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label4.Location = new System.Drawing.Point(7, 195);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(49, 13);
            this.label4.TabIndex = 7;
            this.label4.Text = "Telefono";
            this.label4.Click += new System.EventHandler(this.label4_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label3.Location = new System.Drawing.Point(7, 154);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(52, 13);
            this.label3.TabIndex = 6;
            this.label3.Text = "Direccion";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label2.Location = new System.Drawing.Point(7, 110);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(44, 13);
            this.label2.TabIndex = 5;
            this.label2.Text = "Nombre";
            // 
            // LblDocumento
            // 
            this.LblDocumento.AutoSize = true;
            this.LblDocumento.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.LblDocumento.Location = new System.Drawing.Point(7, 72);
            this.LblDocumento.Name = "LblDocumento";
            this.LblDocumento.Size = new System.Drawing.Size(62, 13);
            this.LblDocumento.TabIndex = 4;
            this.LblDocumento.Text = "Documento";
            // 
            // TxtTelefonoInsertar
            // 
            this.TxtTelefonoInsertar.Location = new System.Drawing.Point(75, 192);
            this.TxtTelefonoInsertar.Name = "TxtTelefonoInsertar";
            this.TxtTelefonoInsertar.Size = new System.Drawing.Size(100, 20);
            this.TxtTelefonoInsertar.TabIndex = 3;
            this.TxtTelefonoInsertar.TextChanged += new System.EventHandler(this.TxtTelefonoInsertar_TextChanged);
            // 
            // TxtDireccionInsertar
            // 
            this.TxtDireccionInsertar.Location = new System.Drawing.Point(75, 147);
            this.TxtDireccionInsertar.Name = "TxtDireccionInsertar";
            this.TxtDireccionInsertar.Size = new System.Drawing.Size(100, 20);
            this.TxtDireccionInsertar.TabIndex = 2;
            // 
            // TxtNombreInsertar
            // 
            this.TxtNombreInsertar.Location = new System.Drawing.Point(75, 103);
            this.TxtNombreInsertar.Name = "TxtNombreInsertar";
            this.TxtNombreInsertar.Size = new System.Drawing.Size(100, 20);
            this.TxtNombreInsertar.TabIndex = 1;
            this.TxtNombreInsertar.TextChanged += new System.EventHandler(this.TxtNombreInsertar_TextChanged);
            // 
            // TxtDocumentoInsertar
            // 
            this.TxtDocumentoInsertar.Location = new System.Drawing.Point(75, 65);
            this.TxtDocumentoInsertar.Name = "TxtDocumentoInsertar";
            this.TxtDocumentoInsertar.Size = new System.Drawing.Size(100, 20);
            this.TxtDocumentoInsertar.TabIndex = 0;
            this.TxtDocumentoInsertar.TextChanged += new System.EventHandler(this.TxtDocumentoInsertar_TextChanged);
            // 
            // TpgModificar
            // 
            this.TpgModificar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpgModificar.Controls.Add(this.BtnEditar);
            this.TpgModificar.Controls.Add(this.DgvEditarClientes);
            this.TpgModificar.Controls.Add(this.BtnConsultarEditarCliente);
            this.TpgModificar.Controls.Add(this.CmbModificarClientes);
            this.TpgModificar.Controls.Add(this.TxtCorreoEditar);
            this.TpgModificar.Controls.Add(this.TxtTelefonoEditar);
            this.TpgModificar.Controls.Add(this.TxtDireccionEditar);
            this.TpgModificar.Controls.Add(this.TxtNombreEditar);
            this.TpgModificar.Controls.Add(this.TxtDocumentoEditar);
            this.TpgModificar.Controls.Add(this.label9);
            this.TpgModificar.Controls.Add(this.label8);
            this.TpgModificar.Controls.Add(this.label7);
            this.TpgModificar.Controls.Add(this.label6);
            this.TpgModificar.Controls.Add(this.label5);
            this.TpgModificar.Location = new System.Drawing.Point(4, 22);
            this.TpgModificar.Name = "TpgModificar";
            this.TpgModificar.Padding = new System.Windows.Forms.Padding(3);
            this.TpgModificar.Size = new System.Drawing.Size(889, 419);
            this.TpgModificar.TabIndex = 2;
            this.TpgModificar.Text = "Modificar";
            // 
            // BtnEditar
            // 
            this.BtnEditar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnEditar.Location = new System.Drawing.Point(187, 285);
            this.BtnEditar.Name = "BtnEditar";
            this.BtnEditar.Size = new System.Drawing.Size(75, 23);
            this.BtnEditar.TabIndex = 14;
            this.BtnEditar.Text = "Editar";
            this.BtnEditar.UseVisualStyleBackColor = false;
            this.BtnEditar.Click += new System.EventHandler(this.BtnEditar_Click);
            // 
            // DgvEditarClientes
            // 
            this.DgvEditarClientes.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.DgvEditarClientes.Location = new System.Drawing.Point(388, 158);
            this.DgvEditarClientes.Name = "DgvEditarClientes";
            this.DgvEditarClientes.Size = new System.Drawing.Size(240, 150);
            this.DgvEditarClientes.TabIndex = 13;
            // 
            // BtnConsultarEditarCliente
            // 
            this.BtnConsultarEditarCliente.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultarEditarCliente.Location = new System.Drawing.Point(443, 94);
            this.BtnConsultarEditarCliente.Name = "BtnConsultarEditarCliente";
            this.BtnConsultarEditarCliente.Size = new System.Drawing.Size(75, 23);
            this.BtnConsultarEditarCliente.TabIndex = 12;
            this.BtnConsultarEditarCliente.Text = "Consultar";
            this.BtnConsultarEditarCliente.UseVisualStyleBackColor = false;
            this.BtnConsultarEditarCliente.Click += new System.EventHandler(this.BtnConsultarEditarCliente_Click);
            // 
            // CmbModificarClientes
            // 
            this.CmbModificarClientes.FormattingEnabled = true;
            this.CmbModificarClientes.Location = new System.Drawing.Point(418, 57);
            this.CmbModificarClientes.Name = "CmbModificarClientes";
            this.CmbModificarClientes.Size = new System.Drawing.Size(121, 21);
            this.CmbModificarClientes.TabIndex = 11;
            // 
            // TxtCorreoEditar
            // 
            this.TxtCorreoEditar.Location = new System.Drawing.Point(88, 230);
            this.TxtCorreoEditar.Name = "TxtCorreoEditar";
            this.TxtCorreoEditar.Size = new System.Drawing.Size(100, 20);
            this.TxtCorreoEditar.TabIndex = 9;
            // 
            // TxtTelefonoEditar
            // 
            this.TxtTelefonoEditar.Location = new System.Drawing.Point(88, 194);
            this.TxtTelefonoEditar.Name = "TxtTelefonoEditar";
            this.TxtTelefonoEditar.Size = new System.Drawing.Size(100, 20);
            this.TxtTelefonoEditar.TabIndex = 8;
            this.TxtTelefonoEditar.TextChanged += new System.EventHandler(this.TxtTelefonoEditar_TextChanged);
            // 
            // TxtDireccionEditar
            // 
            this.TxtDireccionEditar.Location = new System.Drawing.Point(88, 158);
            this.TxtDireccionEditar.Name = "TxtDireccionEditar";
            this.TxtDireccionEditar.Size = new System.Drawing.Size(100, 20);
            this.TxtDireccionEditar.TabIndex = 7;
            // 
            // TxtNombreEditar
            // 
            this.TxtNombreEditar.Location = new System.Drawing.Point(88, 122);
            this.TxtNombreEditar.Name = "TxtNombreEditar";
            this.TxtNombreEditar.Size = new System.Drawing.Size(100, 20);
            this.TxtNombreEditar.TabIndex = 6;
            // 
            // TxtDocumentoEditar
            // 
            this.TxtDocumentoEditar.Location = new System.Drawing.Point(88, 84);
            this.TxtDocumentoEditar.Name = "TxtDocumentoEditar";
            this.TxtDocumentoEditar.ReadOnly = true;
            this.TxtDocumentoEditar.Size = new System.Drawing.Size(100, 20);
            this.TxtDocumentoEditar.TabIndex = 5;
            this.TxtDocumentoEditar.TextChanged += new System.EventHandler(this.TxtDocumentoEditar_TextChanged);
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label9.Location = new System.Drawing.Point(19, 230);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(38, 13);
            this.label9.TabIndex = 4;
            this.label9.Text = "Correo";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label8.Location = new System.Drawing.Point(19, 194);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(49, 13);
            this.label8.TabIndex = 3;
            this.label8.Text = "Telefono";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label7.Location = new System.Drawing.Point(19, 158);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(49, 13);
            this.label7.TabIndex = 2;
            this.label7.Text = "Dirrecion";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label6.Location = new System.Drawing.Point(19, 122);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(44, 13);
            this.label6.TabIndex = 1;
            this.label6.Text = "Nombre";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label5.Location = new System.Drawing.Point(19, 84);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(62, 13);
            this.label5.TabIndex = 0;
            this.label5.Text = "Documento";
            // 
            // TpgEliminar
            // 
            this.TpgEliminar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpgEliminar.Controls.Add(this.BtnEliminar);
            this.TpgEliminar.Controls.Add(this.cmbEliminarClientes);
            this.TpgEliminar.Location = new System.Drawing.Point(4, 22);
            this.TpgEliminar.Name = "TpgEliminar";
            this.TpgEliminar.Padding = new System.Windows.Forms.Padding(3);
            this.TpgEliminar.Size = new System.Drawing.Size(889, 419);
            this.TpgEliminar.TabIndex = 3;
            this.TpgEliminar.Text = "Eliminar";
            // 
            // BtnEliminar
            // 
            this.BtnEliminar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnEliminar.Location = new System.Drawing.Point(323, 57);
            this.BtnEliminar.Name = "BtnEliminar";
            this.BtnEliminar.Size = new System.Drawing.Size(75, 23);
            this.BtnEliminar.TabIndex = 1;
            this.BtnEliminar.Text = "Eliminar";
            this.BtnEliminar.UseVisualStyleBackColor = false;
            this.BtnEliminar.Click += new System.EventHandler(this.BtnEliminar_Click);
            // 
            // cmbEliminarClientes
            // 
            this.cmbEliminarClientes.FormattingEnabled = true;
            this.cmbEliminarClientes.Location = new System.Drawing.Point(304, 19);
            this.cmbEliminarClientes.Name = "cmbEliminarClientes";
            this.cmbEliminarClientes.Size = new System.Drawing.Size(121, 21);
            this.cmbEliminarClientes.TabIndex = 0;
            // 
            // errorProvider1
            // 
            this.errorProvider1.ContainerControl = this;
            // 
            // errorProvider2
            // 
            this.errorProvider2.ContainerControl = this;
            // 
            // Clientes
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(886, 442);
            this.Controls.Add(this.tabControl1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "Clientes";
            this.Text = "Clientes";
            this.Load += new System.EventHandler(this.Clientes_Load);
            this.tabControl1.ResumeLayout(false);
            this.TpgCopnsultar.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.DgvCClientes)).EndInit();
            this.TpgInsertar.ResumeLayout(false);
            this.TpgInsertar.PerformLayout();
            this.TpgModificar.ResumeLayout(false);
            this.TpgModificar.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.DgvEditarClientes)).EndInit();
            this.TpgEliminar.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider2)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage TpgCopnsultar;
        private System.Windows.Forms.TabPage TpgInsertar;
        private System.Windows.Forms.TabPage TpgModificar;
        private System.Windows.Forms.TabPage TpgEliminar;
        private System.Windows.Forms.DataGridView DgvCClientes;
        private System.Windows.Forms.ComboBox cmbCC;
        private System.Windows.Forms.Button BtnConsultar;
        private System.Windows.Forms.Button BtnConsultartodo;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label LblDocumento;
        private System.Windows.Forms.TextBox TxtTelefonoInsertar;
        private System.Windows.Forms.TextBox TxtDireccionInsertar;
        private System.Windows.Forms.TextBox TxtNombreInsertar;
        private System.Windows.Forms.TextBox TxtDocumentoInsertar;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox TxtCorreoInsertar;
        private System.Windows.Forms.Button BtnInsertar;
        private System.Windows.Forms.Button BtnConsultarEditarCliente;
        private System.Windows.Forms.ComboBox CmbModificarClientes;
        private System.Windows.Forms.TextBox TxtCorreoEditar;
        private System.Windows.Forms.TextBox TxtTelefonoEditar;
        private System.Windows.Forms.TextBox TxtDireccionEditar;
        private System.Windows.Forms.TextBox TxtNombreEditar;
        private System.Windows.Forms.TextBox TxtDocumentoEditar;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.DataGridView DgvEditarClientes;
        private System.Windows.Forms.Button BtnEditar;
        private System.Windows.Forms.ComboBox cmbEliminarClientes;
        private System.Windows.Forms.Button BtnEliminar;
        private System.Windows.Forms.Button BtnVolver;
        private System.Windows.Forms.Label LblValidarCorreo;
        private System.Windows.Forms.Label LblVerificvarDocumento;
        private System.Windows.Forms.ErrorProvider errorProvider1;
        private System.Windows.Forms.ErrorProvider errorProvider2;
        private System.Windows.Forms.Label LblVerificarNombre;
    }
}