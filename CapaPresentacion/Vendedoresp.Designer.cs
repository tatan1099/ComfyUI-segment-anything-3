namespace CapaPresentacion
{
    partial class Vendedoresp
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
            this.TPVendedores = new System.Windows.Forms.TabControl();
            this.TpInsertar = new System.Windows.Forms.TabPage();
            this.BtnVolver = new System.Windows.Forms.Button();
            this.BtnMostraiP = new System.Windows.Forms.Button();
            this.DgvInser = new System.Windows.Forms.DataGridView();
            this.BtnGuardar = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.LblN = new System.Windows.Forms.Label();
            this.TxtNombre = new System.Windows.Forms.TextBox();
            this.TxtContrasena = new System.Windows.Forms.TextBox();
            this.TxtUsuario = new System.Windows.Forms.TextBox();
            this.TxtCodigo = new System.Windows.Forms.TextBox();
            this.TPModificar = new System.Windows.Forms.TabPage();
            this.CMBModi = new System.Windows.Forms.ComboBox();
            this.button5 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.Dgvmo = new System.Windows.Forms.DataGridView();
            this.TxtUsuarioE = new System.Windows.Forms.TextBox();
            this.TxtContrasenaE = new System.Windows.Forms.TextBox();
            this.TxtNombreE = new System.Windows.Forms.TextBox();
            this.TxtCodigoe = new System.Windows.Forms.TextBox();
            this.BtnEditar = new System.Windows.Forms.Button();
            this.BtnConsultar = new System.Windows.Forms.Button();
            this.TpEliminar = new System.Windows.Forms.TabPage();
            this.BtnEliminar = new System.Windows.Forms.Button();
            this.CmbEliminar = new System.Windows.Forms.ComboBox();
            this.TpConsultar = new System.Windows.Forms.TabPage();
            this.DgvMostrar = new System.Windows.Forms.DataGridView();
            this.BtnConsultart = new System.Windows.Forms.Button();
            this.BtnConsultar1 = new System.Windows.Forms.Button();
            this.CmMostrar = new System.Windows.Forms.ComboBox();
            this.LblVerificarCodigo = new System.Windows.Forms.Label();
            this.errorProvider1 = new System.Windows.Forms.ErrorProvider(this.components);
            this.TPVendedores.SuspendLayout();
            this.TpInsertar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.DgvInser)).BeginInit();
            this.TPModificar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Dgvmo)).BeginInit();
            this.TpEliminar.SuspendLayout();
            this.TpConsultar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.DgvMostrar)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider1)).BeginInit();
            this.SuspendLayout();
            // 
            // TPVendedores
            // 
            this.TPVendedores.Controls.Add(this.TpInsertar);
            this.TPVendedores.Controls.Add(this.TPModificar);
            this.TPVendedores.Controls.Add(this.TpEliminar);
            this.TPVendedores.Controls.Add(this.TpConsultar);
            this.TPVendedores.Location = new System.Drawing.Point(-2, 0);
            this.TPVendedores.Name = "TPVendedores";
            this.TPVendedores.SelectedIndex = 0;
            this.TPVendedores.Size = new System.Drawing.Size(800, 455);
            this.TPVendedores.TabIndex = 0;
            // 
            // TpInsertar
            // 
            this.TpInsertar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpInsertar.Controls.Add(this.LblVerificarCodigo);
            this.TpInsertar.Controls.Add(this.BtnVolver);
            this.TpInsertar.Controls.Add(this.BtnMostraiP);
            this.TpInsertar.Controls.Add(this.DgvInser);
            this.TpInsertar.Controls.Add(this.BtnGuardar);
            this.TpInsertar.Controls.Add(this.label4);
            this.TpInsertar.Controls.Add(this.label3);
            this.TpInsertar.Controls.Add(this.label2);
            this.TpInsertar.Controls.Add(this.LblN);
            this.TpInsertar.Controls.Add(this.TxtNombre);
            this.TpInsertar.Controls.Add(this.TxtContrasena);
            this.TpInsertar.Controls.Add(this.TxtUsuario);
            this.TpInsertar.Controls.Add(this.TxtCodigo);
            this.TpInsertar.Location = new System.Drawing.Point(4, 22);
            this.TpInsertar.Name = "TpInsertar";
            this.TpInsertar.Padding = new System.Windows.Forms.Padding(3);
            this.TpInsertar.Size = new System.Drawing.Size(792, 429);
            this.TpInsertar.TabIndex = 0;
            this.TpInsertar.Text = "Insertar";
            // 
            // BtnVolver
            // 
            this.BtnVolver.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnVolver.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnVolver.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnVolver.Location = new System.Drawing.Point(678, 297);
            this.BtnVolver.Name = "BtnVolver";
            this.BtnVolver.Size = new System.Drawing.Size(108, 44);
            this.BtnVolver.TabIndex = 12;
            this.BtnVolver.Text = "Volver";
            this.BtnVolver.UseVisualStyleBackColor = false;
            this.BtnVolver.Click += new System.EventHandler(this.BtnVolver_Click);
            // 
            // BtnMostraiP
            // 
            this.BtnMostraiP.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnMostraiP.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnMostraiP.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnMostraiP.Location = new System.Drawing.Point(550, 297);
            this.BtnMostraiP.Name = "BtnMostraiP";
            this.BtnMostraiP.Size = new System.Drawing.Size(108, 44);
            this.BtnMostraiP.TabIndex = 11;
            this.BtnMostraiP.Text = "Mostrar";
            this.BtnMostraiP.UseVisualStyleBackColor = false;
            // 
            // DgvInser
            // 
            this.DgvInser.BackgroundColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.DgvInser.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.DgvInser.GridColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.DgvInser.Location = new System.Drawing.Point(389, 6);
            this.DgvInser.Name = "DgvInser";
            this.DgvInser.Size = new System.Drawing.Size(397, 266);
            this.DgvInser.TabIndex = 9;
            // 
            // BtnGuardar
            // 
            this.BtnGuardar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnGuardar.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnGuardar.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnGuardar.Location = new System.Drawing.Point(422, 297);
            this.BtnGuardar.Name = "BtnGuardar";
            this.BtnGuardar.Size = new System.Drawing.Size(108, 44);
            this.BtnGuardar.TabIndex = 8;
            this.BtnGuardar.Text = "Guardar";
            this.BtnGuardar.UseVisualStyleBackColor = false;
            this.BtnGuardar.Click += new System.EventHandler(this.BtnGuardar_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label4.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label4.Location = new System.Drawing.Point(24, 268);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(86, 25);
            this.label4.TabIndex = 7;
            this.label4.Text = "Nombre";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label3.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label3.Location = new System.Drawing.Point(20, 199);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(118, 25);
            this.label3.TabIndex = 6;
            this.label3.Text = "Contraseña";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label2.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label2.Location = new System.Drawing.Point(17, 122);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(83, 25);
            this.label2.TabIndex = 5;
            this.label2.Text = "Usuario";
            // 
            // LblN
            // 
            this.LblN.AutoSize = true;
            this.LblN.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.LblN.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.LblN.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.LblN.Location = new System.Drawing.Point(24, 56);
            this.LblN.Name = "LblN";
            this.LblN.Size = new System.Drawing.Size(73, 25);
            this.LblN.TabIndex = 4;
            this.LblN.Text = "Codigo";
            // 
            // TxtNombre
            // 
            this.TxtNombre.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtNombre.Location = new System.Drawing.Point(144, 264);
            this.TxtNombre.Name = "TxtNombre";
            this.TxtNombre.Size = new System.Drawing.Size(126, 26);
            this.TxtNombre.TabIndex = 3;
            this.TxtNombre.TextChanged += new System.EventHandler(this.TxtNombre_TextChanged);
            // 
            // TxtContrasena
            // 
            this.TxtContrasena.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtContrasena.Location = new System.Drawing.Point(144, 198);
            this.TxtContrasena.Name = "TxtContrasena";
            this.TxtContrasena.Size = new System.Drawing.Size(126, 26);
            this.TxtContrasena.TabIndex = 2;
            // 
            // TxtUsuario
            // 
            this.TxtUsuario.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtUsuario.Location = new System.Drawing.Point(141, 124);
            this.TxtUsuario.Name = "TxtUsuario";
            this.TxtUsuario.Size = new System.Drawing.Size(126, 26);
            this.TxtUsuario.TabIndex = 1;
            // 
            // TxtCodigo
            // 
            this.TxtCodigo.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtCodigo.Location = new System.Drawing.Point(141, 58);
            this.TxtCodigo.Name = "TxtCodigo";
            this.TxtCodigo.Size = new System.Drawing.Size(129, 26);
            this.TxtCodigo.TabIndex = 0;
            this.TxtCodigo.TextChanged += new System.EventHandler(this.TxtCodigo_TextChanged);
            // 
            // TPModificar
            // 
            this.TPModificar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TPModificar.Controls.Add(this.CMBModi);
            this.TPModificar.Controls.Add(this.button5);
            this.TPModificar.Controls.Add(this.button4);
            this.TPModificar.Controls.Add(this.button3);
            this.TPModificar.Controls.Add(this.button1);
            this.TPModificar.Controls.Add(this.button2);
            this.TPModificar.Controls.Add(this.Dgvmo);
            this.TPModificar.Controls.Add(this.TxtUsuarioE);
            this.TPModificar.Controls.Add(this.TxtContrasenaE);
            this.TPModificar.Controls.Add(this.TxtNombreE);
            this.TPModificar.Controls.Add(this.TxtCodigoe);
            this.TPModificar.Controls.Add(this.BtnEditar);
            this.TPModificar.Controls.Add(this.BtnConsultar);
            this.TPModificar.Location = new System.Drawing.Point(4, 22);
            this.TPModificar.Name = "TPModificar";
            this.TPModificar.Padding = new System.Windows.Forms.Padding(3);
            this.TPModificar.Size = new System.Drawing.Size(792, 429);
            this.TPModificar.TabIndex = 1;
            this.TPModificar.Text = "Modificar";
            // 
            // CMBModi
            // 
            this.CMBModi.FormattingEnabled = true;
            this.CMBModi.Location = new System.Drawing.Point(506, 33);
            this.CMBModi.Name = "CMBModi";
            this.CMBModi.Size = new System.Drawing.Size(121, 21);
            this.CMBModi.TabIndex = 13;
            // 
            // button5
            // 
            this.button5.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.button5.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button5.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button5.Location = new System.Drawing.Point(10, 259);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(130, 38);
            this.button5.TabIndex = 12;
            this.button5.Text = "Nombre";
            this.button5.UseVisualStyleBackColor = false;
            // 
            // button4
            // 
            this.button4.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.button4.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button4.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button4.Location = new System.Drawing.Point(10, 202);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(130, 38);
            this.button4.TabIndex = 11;
            this.button4.Text = "Contraseña";
            this.button4.UseVisualStyleBackColor = false;
            // 
            // button3
            // 
            this.button3.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.button3.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button3.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button3.Location = new System.Drawing.Point(10, 144);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(119, 38);
            this.button3.TabIndex = 10;
            this.button3.Text = "Usuario";
            this.button3.UseVisualStyleBackColor = false;
            // 
            // button1
            // 
            this.button1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.button1.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button1.Location = new System.Drawing.Point(10, 79);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(119, 47);
            this.button1.TabIndex = 9;
            this.button1.Text = "Codigo";
            this.button1.UseVisualStyleBackColor = false;
            // 
            // button2
            // 
            this.button2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.button2.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button2.Location = new System.Drawing.Point(395, 22);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(96, 38);
            this.button2.TabIndex = 8;
            this.button2.Text = "Modificar";
            this.button2.UseVisualStyleBackColor = false;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // Dgvmo
            // 
            this.Dgvmo.BackgroundColor = System.Drawing.SystemColors.ButtonHighlight;
            this.Dgvmo.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.Dgvmo.Location = new System.Drawing.Point(357, 92);
            this.Dgvmo.Name = "Dgvmo";
            this.Dgvmo.Size = new System.Drawing.Size(429, 324);
            this.Dgvmo.TabIndex = 7;
            // 
            // TxtUsuarioE
            // 
            this.TxtUsuarioE.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtUsuarioE.ForeColor = System.Drawing.Color.Navy;
            this.TxtUsuarioE.Location = new System.Drawing.Point(158, 144);
            this.TxtUsuarioE.Name = "TxtUsuarioE";
            this.TxtUsuarioE.Size = new System.Drawing.Size(121, 26);
            this.TxtUsuarioE.TabIndex = 6;
            // 
            // TxtContrasenaE
            // 
            this.TxtContrasenaE.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtContrasenaE.Location = new System.Drawing.Point(158, 202);
            this.TxtContrasenaE.Name = "TxtContrasenaE";
            this.TxtContrasenaE.Size = new System.Drawing.Size(121, 26);
            this.TxtContrasenaE.TabIndex = 5;
            this.TxtContrasenaE.TextChanged += new System.EventHandler(this.TxtContrasenaE_TextChanged);
            // 
            // TxtNombreE
            // 
            this.TxtNombreE.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtNombreE.Location = new System.Drawing.Point(158, 259);
            this.TxtNombreE.Name = "TxtNombreE";
            this.TxtNombreE.Size = new System.Drawing.Size(121, 26);
            this.TxtNombreE.TabIndex = 4;
            this.TxtNombreE.TextChanged += new System.EventHandler(this.TxtNombreE_TextChanged);
            // 
            // TxtCodigoe
            // 
            this.TxtCodigoe.Enabled = false;
            this.TxtCodigoe.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtCodigoe.ForeColor = System.Drawing.Color.Navy;
            this.TxtCodigoe.Location = new System.Drawing.Point(158, 83);
            this.TxtCodigoe.Name = "TxtCodigoe";
            this.TxtCodigoe.Size = new System.Drawing.Size(121, 26);
            this.TxtCodigoe.TabIndex = 3;
            // 
            // BtnEditar
            // 
            this.BtnEditar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnEditar.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnEditar.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnEditar.Location = new System.Drawing.Point(110, 322);
            this.BtnEditar.Name = "BtnEditar";
            this.BtnEditar.Size = new System.Drawing.Size(152, 54);
            this.BtnEditar.TabIndex = 2;
            this.BtnEditar.Text = "Guardar Cambios";
            this.BtnEditar.UseVisualStyleBackColor = false;
            this.BtnEditar.Click += new System.EventHandler(this.BtnEditar_Click);
            // 
            // BtnConsultar
            // 
            this.BtnConsultar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultar.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnConsultar.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnConsultar.Location = new System.Drawing.Point(633, 22);
            this.BtnConsultar.Name = "BtnConsultar";
            this.BtnConsultar.Size = new System.Drawing.Size(96, 38);
            this.BtnConsultar.TabIndex = 1;
            this.BtnConsultar.Text = "Consultar";
            this.BtnConsultar.UseVisualStyleBackColor = false;
            this.BtnConsultar.Click += new System.EventHandler(this.BtnConsultar_Click);
            // 
            // TpEliminar
            // 
            this.TpEliminar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpEliminar.Controls.Add(this.BtnEliminar);
            this.TpEliminar.Controls.Add(this.CmbEliminar);
            this.TpEliminar.Location = new System.Drawing.Point(4, 22);
            this.TpEliminar.Name = "TpEliminar";
            this.TpEliminar.Padding = new System.Windows.Forms.Padding(3);
            this.TpEliminar.Size = new System.Drawing.Size(792, 429);
            this.TpEliminar.TabIndex = 2;
            this.TpEliminar.Text = "Eliminar";
            // 
            // BtnEliminar
            // 
            this.BtnEliminar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnEliminar.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnEliminar.Location = new System.Drawing.Point(311, 101);
            this.BtnEliminar.Name = "BtnEliminar";
            this.BtnEliminar.Size = new System.Drawing.Size(110, 47);
            this.BtnEliminar.TabIndex = 1;
            this.BtnEliminar.Text = "Eliminar";
            this.BtnEliminar.UseVisualStyleBackColor = false;
            this.BtnEliminar.Click += new System.EventHandler(this.BtnEliminar_Click);
            // 
            // CmbEliminar
            // 
            this.CmbEliminar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.CmbEliminar.Font = new System.Drawing.Font("Microsoft Sans Serif", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.CmbEliminar.ForeColor = System.Drawing.Color.Black;
            this.CmbEliminar.FormattingEnabled = true;
            this.CmbEliminar.Location = new System.Drawing.Point(259, 39);
            this.CmbEliminar.Name = "CmbEliminar";
            this.CmbEliminar.Size = new System.Drawing.Size(187, 32);
            this.CmbEliminar.TabIndex = 0;
            // 
            // TpConsultar
            // 
            this.TpConsultar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpConsultar.Controls.Add(this.DgvMostrar);
            this.TpConsultar.Controls.Add(this.BtnConsultart);
            this.TpConsultar.Controls.Add(this.BtnConsultar1);
            this.TpConsultar.Controls.Add(this.CmMostrar);
            this.TpConsultar.Location = new System.Drawing.Point(4, 22);
            this.TpConsultar.Name = "TpConsultar";
            this.TpConsultar.Padding = new System.Windows.Forms.Padding(3);
            this.TpConsultar.Size = new System.Drawing.Size(792, 429);
            this.TpConsultar.TabIndex = 3;
            this.TpConsultar.Text = "Consultar";
            // 
            // DgvMostrar
            // 
            this.DgvMostrar.BackgroundColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.DgvMostrar.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.DgvMostrar.GridColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.DgvMostrar.Location = new System.Drawing.Point(6, 198);
            this.DgvMostrar.Name = "DgvMostrar";
            this.DgvMostrar.Size = new System.Drawing.Size(769, 218);
            this.DgvMostrar.TabIndex = 3;
            // 
            // BtnConsultart
            // 
            this.BtnConsultart.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultart.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnConsultart.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnConsultart.Location = new System.Drawing.Point(175, 107);
            this.BtnConsultart.Name = "BtnConsultart";
            this.BtnConsultart.Size = new System.Drawing.Size(143, 35);
            this.BtnConsultart.TabIndex = 2;
            this.BtnConsultart.Text = "Mostrar Todo";
            this.BtnConsultart.UseVisualStyleBackColor = false;
            this.BtnConsultart.Click += new System.EventHandler(this.BtnConsultart_Click);
            // 
            // BtnConsultar1
            // 
            this.BtnConsultar1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultar1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnConsultar1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnConsultar1.Location = new System.Drawing.Point(466, 109);
            this.BtnConsultar1.Name = "BtnConsultar1";
            this.BtnConsultar1.Size = new System.Drawing.Size(105, 31);
            this.BtnConsultar1.TabIndex = 1;
            this.BtnConsultar1.Text = "Consultar";
            this.BtnConsultar1.UseVisualStyleBackColor = false;
            this.BtnConsultar1.Click += new System.EventHandler(this.BtnConsultar1_Click);
            // 
            // CmMostrar
            // 
            this.CmMostrar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.CmMostrar.Font = new System.Drawing.Font("Microsoft Sans Serif", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.CmMostrar.FormattingEnabled = true;
            this.CmMostrar.Location = new System.Drawing.Point(175, 55);
            this.CmMostrar.Name = "CmMostrar";
            this.CmMostrar.Size = new System.Drawing.Size(396, 32);
            this.CmMostrar.TabIndex = 0;
            // 
            // LblVerificarCodigo
            // 
            this.LblVerificarCodigo.AutoSize = true;
            this.LblVerificarCodigo.Location = new System.Drawing.Point(287, 68);
            this.LblVerificarCodigo.Name = "LblVerificarCodigo";
            this.LblVerificarCodigo.Size = new System.Drawing.Size(0, 13);
            this.LblVerificarCodigo.TabIndex = 13;
            // 
            // errorProvider1
            // 
            this.errorProvider1.ContainerControl = this;
            // 
            // Vendedoresp
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.TPVendedores);
            this.Name = "Vendedoresp";
            this.Text = "Vendedoresp";
            this.Load += new System.EventHandler(this.Vendedoresp_Load);
            this.TPVendedores.ResumeLayout(false);
            this.TpInsertar.ResumeLayout(false);
            this.TpInsertar.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.DgvInser)).EndInit();
            this.TPModificar.ResumeLayout(false);
            this.TPModificar.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Dgvmo)).EndInit();
            this.TpEliminar.ResumeLayout(false);
            this.TpConsultar.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.DgvMostrar)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl TPVendedores;
        private System.Windows.Forms.TabPage TpInsertar;
        private System.Windows.Forms.Button BtnGuardar;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label LblN;
        private System.Windows.Forms.TextBox TxtNombre;
        private System.Windows.Forms.TextBox TxtContrasena;
        private System.Windows.Forms.TextBox TxtUsuario;
        private System.Windows.Forms.TextBox TxtCodigo;
        private System.Windows.Forms.TabPage TPModificar;
        private System.Windows.Forms.TabPage TpEliminar;
        private System.Windows.Forms.TabPage TpConsultar;
        private System.Windows.Forms.TextBox TxtUsuarioE;
        private System.Windows.Forms.TextBox TxtContrasenaE;
        private System.Windows.Forms.TextBox TxtNombreE;
        private System.Windows.Forms.TextBox TxtCodigoe;
        private System.Windows.Forms.Button BtnEditar;
        private System.Windows.Forms.Button BtnConsultar;
        private System.Windows.Forms.Button BtnEliminar;
        private System.Windows.Forms.ComboBox CmbEliminar;
        private System.Windows.Forms.Button BtnConsultart;
        private System.Windows.Forms.Button BtnConsultar1;
        private System.Windows.Forms.ComboBox CmMostrar;
        private System.Windows.Forms.DataGridView DgvMostrar;
        private System.Windows.Forms.DataGridView Dgvmo;
        private System.Windows.Forms.DataGridView DgvInser;
        private System.Windows.Forms.Button BtnMostraiP;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.ComboBox CMBModi;
        private System.Windows.Forms.Button BtnVolver;
        private System.Windows.Forms.Label LblVerificarCodigo;
        private System.Windows.Forms.ErrorProvider errorProvider1;
    }
}