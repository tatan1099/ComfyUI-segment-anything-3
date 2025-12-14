namespace CapaPresentacion
{
    partial class Productos
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
            this.TpgConsultar = new System.Windows.Forms.TabPage();
            this.button4 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.BtnConsultar = new System.Windows.Forms.Button();
            this.CmbMp = new System.Windows.Forms.ComboBox();
            this.DgvMp = new System.Windows.Forms.DataGridView();
            this.TpgEditar = new System.Windows.Forms.TabPage();
            this.TxtCodN = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.Mostra = new System.Windows.Forms.Button();
            this.BtnEditar = new System.Windows.Forms.Button();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.TxtCANTIDADE = new System.Windows.Forms.TextBox();
            this.TxtValorUE = new System.Windows.Forms.TextBox();
            this.TxtDescripcionE = new System.Windows.Forms.TextBox();
            this.TxtCodigoE = new System.Windows.Forms.TextBox();
            this.BtnConsultarE = new System.Windows.Forms.Button();
            this.CmbEdi = new System.Windows.Forms.ComboBox();
            this.TpgEliminar = new System.Windows.Forms.TabPage();
            this.BtnEliminar = new System.Windows.Forms.Button();
            this.CmbEli = new System.Windows.Forms.ComboBox();
            this.TPGInsertar = new System.Windows.Forms.TabPage();
            this.TxtCodigo = new System.Windows.Forms.TextBox();
            this.TxtDescripcion = new System.Windows.Forms.TextBox();
            this.TxtValorunitario = new System.Windows.Forms.TextBox();
            this.TxtCantidad = new System.Windows.Forms.TextBox();
            this.BtnInsertar = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.LblVerificar = new System.Windows.Forms.Label();
            this.errorProvider1 = new System.Windows.Forms.ErrorProvider(this.components);
            this.errorProvider2 = new System.Windows.Forms.ErrorProvider(this.components);
            this.LblDescriVerificar = new System.Windows.Forms.Label();
            this.LblValorVeri = new System.Windows.Forms.Label();
            this.LblCantiDADvERI = new System.Windows.Forms.Label();
            this.tabControl1.SuspendLayout();
            this.TpgConsultar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.DgvMp)).BeginInit();
            this.TpgEditar.SuspendLayout();
            this.TpgEliminar.SuspendLayout();
            this.TPGInsertar.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider2)).BeginInit();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.TpgConsultar);
            this.tabControl1.Controls.Add(this.TpgEditar);
            this.tabControl1.Controls.Add(this.TpgEliminar);
            this.tabControl1.Controls.Add(this.TPGInsertar);
            this.tabControl1.Location = new System.Drawing.Point(-2, -1);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(799, 439);
            this.tabControl1.TabIndex = 29;
            // 
            // TpgConsultar
            // 
            this.TpgConsultar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpgConsultar.Controls.Add(this.button4);
            this.TpgConsultar.Controls.Add(this.button3);
            this.TpgConsultar.Controls.Add(this.button2);
            this.TpgConsultar.Controls.Add(this.BtnConsultar);
            this.TpgConsultar.Controls.Add(this.CmbMp);
            this.TpgConsultar.Controls.Add(this.DgvMp);
            this.TpgConsultar.Location = new System.Drawing.Point(4, 22);
            this.TpgConsultar.Name = "TpgConsultar";
            this.TpgConsultar.Padding = new System.Windows.Forms.Padding(3);
            this.TpgConsultar.Size = new System.Drawing.Size(791, 413);
            this.TpgConsultar.TabIndex = 1;
            this.TpgConsultar.Text = "Consultar";
            this.TpgConsultar.Click += new System.EventHandler(this.TpgConsultar_Click);
            // 
            // button4
            // 
            this.button4.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.button4.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button4.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button4.Location = new System.Drawing.Point(617, 52);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(169, 40);
            this.button4.TabIndex = 5;
            this.button4.Text = "Exportar pdf";
            this.button4.UseVisualStyleBackColor = false;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // button3
            // 
            this.button3.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.button3.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button3.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button3.Location = new System.Drawing.Point(617, 6);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(169, 40);
            this.button3.TabIndex = 4;
            this.button3.Text = "Exportar Excel";
            this.button3.UseVisualStyleBackColor = false;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // button2
            // 
            this.button2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.button2.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button2.Location = new System.Drawing.Point(373, 105);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(169, 40);
            this.button2.TabIndex = 3;
            this.button2.Text = "mostrar todo";
            this.button2.UseVisualStyleBackColor = false;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // BtnConsultar
            // 
            this.BtnConsultar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultar.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnConsultar.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnConsultar.Location = new System.Drawing.Point(189, 105);
            this.BtnConsultar.Name = "BtnConsultar";
            this.BtnConsultar.Size = new System.Drawing.Size(169, 40);
            this.BtnConsultar.TabIndex = 2;
            this.BtnConsultar.Text = "Consultar";
            this.BtnConsultar.UseVisualStyleBackColor = false;
            this.BtnConsultar.Click += new System.EventHandler(this.BtnConsultar_Click);
            // 
            // CmbMp
            // 
            this.CmbMp.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.CmbMp.FormattingEnabled = true;
            this.CmbMp.Location = new System.Drawing.Point(228, 38);
            this.CmbMp.Name = "CmbMp";
            this.CmbMp.Size = new System.Drawing.Size(251, 33);
            this.CmbMp.TabIndex = 1;
            // 
            // DgvMp
            // 
            this.DgvMp.BackgroundColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.DgvMp.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.DgvMp.GridColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.DgvMp.Location = new System.Drawing.Point(8, 198);
            this.DgvMp.Name = "DgvMp";
            this.DgvMp.Size = new System.Drawing.Size(783, 197);
            this.DgvMp.TabIndex = 0;
            // 
            // TpgEditar
            // 
            this.TpgEditar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpgEditar.Controls.Add(this.TxtCodN);
            this.TpgEditar.Controls.Add(this.label10);
            this.TpgEditar.Controls.Add(this.Mostra);
            this.TpgEditar.Controls.Add(this.BtnEditar);
            this.TpgEditar.Controls.Add(this.label9);
            this.TpgEditar.Controls.Add(this.label8);
            this.TpgEditar.Controls.Add(this.label7);
            this.TpgEditar.Controls.Add(this.label6);
            this.TpgEditar.Controls.Add(this.label5);
            this.TpgEditar.Controls.Add(this.TxtCANTIDADE);
            this.TpgEditar.Controls.Add(this.TxtValorUE);
            this.TpgEditar.Controls.Add(this.TxtDescripcionE);
            this.TpgEditar.Controls.Add(this.TxtCodigoE);
            this.TpgEditar.Controls.Add(this.BtnConsultarE);
            this.TpgEditar.Controls.Add(this.CmbEdi);
            this.TpgEditar.Location = new System.Drawing.Point(4, 22);
            this.TpgEditar.Name = "TpgEditar";
            this.TpgEditar.Padding = new System.Windows.Forms.Padding(3);
            this.TpgEditar.Size = new System.Drawing.Size(791, 413);
            this.TpgEditar.TabIndex = 2;
            this.TpgEditar.Text = "Editar";
            this.TpgEditar.Click += new System.EventHandler(this.TpgEditar_Click);
            // 
            // TxtCodN
            // 
            this.TxtCodN.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtCodN.Location = new System.Drawing.Point(676, 56);
            this.TxtCodN.Name = "TxtCodN";
            this.TxtCodN.Size = new System.Drawing.Size(100, 20);
            this.TxtCodN.TabIndex = 14;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label10.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label10.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label10.Location = new System.Drawing.Point(686, 25);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(79, 28);
            this.label10.TabIndex = 13;
            this.label10.Text = "Codigo";
            // 
            // Mostra
            // 
            this.Mostra.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.Mostra.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Mostra.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.Mostra.Location = new System.Drawing.Point(419, 71);
            this.Mostra.Name = "Mostra";
            this.Mostra.Size = new System.Drawing.Size(113, 45);
            this.Mostra.TabIndex = 12;
            this.Mostra.Text = "Mostrar Todo";
            this.Mostra.UseVisualStyleBackColor = false;
            this.Mostra.Click += new System.EventHandler(this.Mostra_Click);
            // 
            // BtnEditar
            // 
            this.BtnEditar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnEditar.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnEditar.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnEditar.Location = new System.Drawing.Point(297, 338);
            this.BtnEditar.Name = "BtnEditar";
            this.BtnEditar.Size = new System.Drawing.Size(170, 38);
            this.BtnEditar.TabIndex = 11;
            this.BtnEditar.Text = "Guardar Cambios";
            this.BtnEditar.UseVisualStyleBackColor = false;
            this.BtnEditar.Click += new System.EventHandler(this.BtnEditar_Click);
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label9.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label9.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label9.Location = new System.Drawing.Point(429, 261);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(103, 28);
            this.label9.TabIndex = 10;
            this.label9.Text = "Cantidad";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label8.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label8.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label8.Location = new System.Drawing.Point(50, 261);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(163, 28);
            this.label8.TabIndex = 9;
            this.label8.Text = "Valor Unitario";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label7.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label7.Location = new System.Drawing.Point(166, 268);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(0, 13);
            this.label7.TabIndex = 8;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label6.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label6.Location = new System.Drawing.Point(429, 149);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(129, 28);
            this.label6.TabIndex = 7;
            this.label6.Text = "Descripcion";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label5.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label5.Location = new System.Drawing.Point(50, 157);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(79, 28);
            this.label5.TabIndex = 6;
            this.label5.Text = "Codigo";
            // 
            // TxtCANTIDADE
            // 
            this.TxtCANTIDADE.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtCANTIDADE.ForeColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.TxtCANTIDADE.Location = new System.Drawing.Point(608, 261);
            this.TxtCANTIDADE.Name = "TxtCANTIDADE";
            this.TxtCANTIDADE.Size = new System.Drawing.Size(100, 20);
            this.TxtCANTIDADE.TabIndex = 5;
            this.TxtCANTIDADE.TextChanged += new System.EventHandler(this.TxtCANTIDADE_TextChanged);
            // 
            // TxtValorUE
            // 
            this.TxtValorUE.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtValorUE.ForeColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.TxtValorUE.Location = new System.Drawing.Point(238, 265);
            this.TxtValorUE.Name = "TxtValorUE";
            this.TxtValorUE.Size = new System.Drawing.Size(100, 20);
            this.TxtValorUE.TabIndex = 4;
            this.TxtValorUE.TextChanged += new System.EventHandler(this.TxtValorUE_TextChanged);
            // 
            // TxtDescripcionE
            // 
            this.TxtDescripcionE.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtDescripcionE.ForeColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.TxtDescripcionE.Location = new System.Drawing.Point(608, 149);
            this.TxtDescripcionE.Name = "TxtDescripcionE";
            this.TxtDescripcionE.Size = new System.Drawing.Size(100, 20);
            this.TxtDescripcionE.TabIndex = 3;
            this.TxtDescripcionE.TextChanged += new System.EventHandler(this.TxtDescripcionE_TextChanged);
            // 
            // TxtCodigoE
            // 
            this.TxtCodigoE.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtCodigoE.ForeColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.TxtCodigoE.Location = new System.Drawing.Point(238, 157);
            this.TxtCodigoE.Name = "TxtCodigoE";
            this.TxtCodigoE.ReadOnly = true;
            this.TxtCodigoE.Size = new System.Drawing.Size(100, 20);
            this.TxtCodigoE.TabIndex = 2;
            this.TxtCodigoE.EnabledChanged += new System.EventHandler(this.TxtCodigoE_EnabledChanged);
            this.TxtCodigoE.TextChanged += new System.EventHandler(this.TxtCodigoE_TextChanged);
            // 
            // BtnConsultarE
            // 
            this.BtnConsultarE.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnConsultarE.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnConsultarE.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnConsultarE.Location = new System.Drawing.Point(253, 71);
            this.BtnConsultarE.Name = "BtnConsultarE";
            this.BtnConsultarE.Size = new System.Drawing.Size(113, 45);
            this.BtnConsultarE.TabIndex = 1;
            this.BtnConsultarE.Text = "Consultar";
            this.BtnConsultarE.UseVisualStyleBackColor = false;
            this.BtnConsultarE.Click += new System.EventHandler(this.BtnConsultarE_Click);
            // 
            // CmbEdi
            // 
            this.CmbEdi.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.CmbEdi.Font = new System.Drawing.Font("Microsoft Sans Serif", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.CmbEdi.FormattingEnabled = true;
            this.CmbEdi.Location = new System.Drawing.Point(262, 25);
            this.CmbEdi.Name = "CmbEdi";
            this.CmbEdi.Size = new System.Drawing.Size(254, 32);
            this.CmbEdi.TabIndex = 0;
            this.CmbEdi.SelectedIndexChanged += new System.EventHandler(this.CmbEdi_SelectedIndexChanged);
            // 
            // TpgEliminar
            // 
            this.TpgEliminar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TpgEliminar.Controls.Add(this.BtnEliminar);
            this.TpgEliminar.Controls.Add(this.CmbEli);
            this.TpgEliminar.Location = new System.Drawing.Point(4, 22);
            this.TpgEliminar.Name = "TpgEliminar";
            this.TpgEliminar.Padding = new System.Windows.Forms.Padding(3);
            this.TpgEliminar.Size = new System.Drawing.Size(791, 413);
            this.TpgEliminar.TabIndex = 3;
            this.TpgEliminar.Text = "Eliminar";
            // 
            // BtnEliminar
            // 
            this.BtnEliminar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnEliminar.ForeColor = System.Drawing.Color.Navy;
            this.BtnEliminar.Location = new System.Drawing.Point(303, 103);
            this.BtnEliminar.Name = "BtnEliminar";
            this.BtnEliminar.Size = new System.Drawing.Size(109, 49);
            this.BtnEliminar.TabIndex = 1;
            this.BtnEliminar.Text = "Eliminar";
            this.BtnEliminar.UseVisualStyleBackColor = false;
            this.BtnEliminar.Click += new System.EventHandler(this.BtnEliminar_Click);
            // 
            // CmbEli
            // 
            this.CmbEli.Font = new System.Drawing.Font("Microsoft Sans Serif", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.CmbEli.FormattingEnabled = true;
            this.CmbEli.Location = new System.Drawing.Point(280, 54);
            this.CmbEli.Name = "CmbEli";
            this.CmbEli.Size = new System.Drawing.Size(171, 32);
            this.CmbEli.TabIndex = 0;
            // 
            // TPGInsertar
            // 
            this.TPGInsertar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.TPGInsertar.Controls.Add(this.LblCantiDADvERI);
            this.TPGInsertar.Controls.Add(this.LblValorVeri);
            this.TPGInsertar.Controls.Add(this.LblDescriVerificar);
            this.TPGInsertar.Controls.Add(this.LblVerificar);
            this.TPGInsertar.Controls.Add(this.label4);
            this.TPGInsertar.Controls.Add(this.label3);
            this.TPGInsertar.Controls.Add(this.label2);
            this.TPGInsertar.Controls.Add(this.label1);
            this.TPGInsertar.Controls.Add(this.BtnInsertar);
            this.TPGInsertar.Controls.Add(this.TxtCantidad);
            this.TPGInsertar.Controls.Add(this.TxtValorunitario);
            this.TPGInsertar.Controls.Add(this.TxtDescripcion);
            this.TPGInsertar.Controls.Add(this.TxtCodigo);
            this.TPGInsertar.Location = new System.Drawing.Point(4, 22);
            this.TPGInsertar.Name = "TPGInsertar";
            this.TPGInsertar.Padding = new System.Windows.Forms.Padding(3);
            this.TPGInsertar.Size = new System.Drawing.Size(791, 413);
            this.TPGInsertar.TabIndex = 4;
            this.TPGInsertar.Text = "Insertar";
            // 
            // TxtCodigo
            // 
            this.TxtCodigo.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtCodigo.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtCodigo.Location = new System.Drawing.Point(253, 83);
            this.TxtCodigo.Name = "TxtCodigo";
            this.TxtCodigo.Size = new System.Drawing.Size(252, 38);
            this.TxtCodigo.TabIndex = 0;
            this.TxtCodigo.TextChanged += new System.EventHandler(this.TxtCodigo_TextChanged);
            // 
            // TxtDescripcion
            // 
            this.TxtDescripcion.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtDescripcion.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtDescripcion.Location = new System.Drawing.Point(253, 132);
            this.TxtDescripcion.Name = "TxtDescripcion";
            this.TxtDescripcion.Size = new System.Drawing.Size(252, 38);
            this.TxtDescripcion.TabIndex = 1;
            this.TxtDescripcion.TextChanged += new System.EventHandler(this.TxtDescripcion_TextChanged);
            // 
            // TxtValorunitario
            // 
            this.TxtValorunitario.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtValorunitario.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtValorunitario.Location = new System.Drawing.Point(253, 196);
            this.TxtValorunitario.Name = "TxtValorunitario";
            this.TxtValorunitario.Size = new System.Drawing.Size(252, 38);
            this.TxtValorunitario.TabIndex = 2;
            this.TxtValorunitario.TextChanged += new System.EventHandler(this.TxtValorunitario_TextChanged);
            // 
            // TxtCantidad
            // 
            this.TxtCantidad.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.TxtCantidad.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TxtCantidad.Location = new System.Drawing.Point(253, 271);
            this.TxtCantidad.Name = "TxtCantidad";
            this.TxtCantidad.Size = new System.Drawing.Size(252, 38);
            this.TxtCantidad.TabIndex = 3;
            this.TxtCantidad.TextChanged += new System.EventHandler(this.TxtCantidad_TextChanged);
            // 
            // BtnInsertar
            // 
            this.BtnInsertar.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnInsertar.Font = new System.Drawing.Font("Microsoft Sans Serif", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnInsertar.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnInsertar.Location = new System.Drawing.Point(568, 325);
            this.BtnInsertar.Name = "BtnInsertar";
            this.BtnInsertar.Size = new System.Drawing.Size(91, 50);
            this.BtnInsertar.TabIndex = 4;
            this.BtnInsertar.Text = "Insertar";
            this.BtnInsertar.UseVisualStyleBackColor = false;
            this.BtnInsertar.Click += new System.EventHandler(this.BtnInsertar_Click_1);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label1.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label1.Location = new System.Drawing.Point(92, 86);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(73, 25);
            this.label1.TabIndex = 5;
            this.label1.Text = "Codigo";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label2.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label2.Location = new System.Drawing.Point(92, 212);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(133, 25);
            this.label2.TabIndex = 6;
            this.label2.Text = "Valor unidad";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label3.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label3.Location = new System.Drawing.Point(92, 145);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(119, 25);
            this.label3.TabIndex = 7;
            this.label3.Text = "Descripcion";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(109)))), ((int)(((byte)(169)))), ((int)(((byte)(228)))));
            this.label4.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.label4.Location = new System.Drawing.Point(92, 284);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(95, 25);
            this.label4.TabIndex = 8;
            this.label4.Text = "Cantidad";
            // 
            // LblVerificar
            // 
            this.LblVerificar.AutoSize = true;
            this.LblVerificar.Location = new System.Drawing.Point(531, 97);
            this.LblVerificar.Name = "LblVerificar";
            this.LblVerificar.Size = new System.Drawing.Size(0, 13);
            this.LblVerificar.TabIndex = 9;
            // 
            // errorProvider1
            // 
            this.errorProvider1.ContainerControl = this;
            // 
            // errorProvider2
            // 
            this.errorProvider2.ContainerControl = this;
            // 
            // LblDescriVerificar
            // 
            this.LblDescriVerificar.AutoSize = true;
            this.LblDescriVerificar.Location = new System.Drawing.Point(525, 145);
            this.LblDescriVerificar.Name = "LblDescriVerificar";
            this.LblDescriVerificar.Size = new System.Drawing.Size(0, 13);
            this.LblDescriVerificar.TabIndex = 10;
            // 
            // LblValorVeri
            // 
            this.LblValorVeri.AutoSize = true;
            this.LblValorVeri.Location = new System.Drawing.Point(525, 212);
            this.LblValorVeri.Name = "LblValorVeri";
            this.LblValorVeri.Size = new System.Drawing.Size(0, 13);
            this.LblValorVeri.TabIndex = 11;
            // 
            // LblCantiDADvERI
            // 
            this.LblCantiDADvERI.AutoSize = true;
            this.LblCantiDADvERI.Location = new System.Drawing.Point(534, 284);
            this.LblCantiDADvERI.Name = "LblCantiDADvERI";
            this.LblCantiDADvERI.Size = new System.Drawing.Size(0, 13);
            this.LblCantiDADvERI.TabIndex = 12;
            // 
            // Productos
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(789, 432);
            this.Controls.Add(this.tabControl1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "Productos";
            this.Text = "Productos";
            this.Load += new System.EventHandler(this.Productos_Load);
            this.tabControl1.ResumeLayout(false);
            this.TpgConsultar.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.DgvMp)).EndInit();
            this.TpgEditar.ResumeLayout(false);
            this.TpgEditar.PerformLayout();
            this.TpgEliminar.ResumeLayout(false);
            this.TPGInsertar.ResumeLayout(false);
            this.TPGInsertar.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.errorProvider2)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage TpgConsultar;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button BtnConsultar;
        private System.Windows.Forms.ComboBox CmbMp;
        private System.Windows.Forms.DataGridView DgvMp;
        private System.Windows.Forms.TabPage TpgEditar;
        private System.Windows.Forms.Button BtnEditar;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox TxtCANTIDADE;
        private System.Windows.Forms.TextBox TxtValorUE;
        private System.Windows.Forms.TextBox TxtDescripcionE;
        private System.Windows.Forms.TextBox TxtCodigoE;
        private System.Windows.Forms.Button BtnConsultarE;
        private System.Windows.Forms.ComboBox CmbEdi;
        private System.Windows.Forms.Button Mostra;
        private System.Windows.Forms.TextBox TxtCodN;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TabPage TpgEliminar;
        private System.Windows.Forms.Button BtnEliminar;
        private System.Windows.Forms.ComboBox CmbEli;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.TabPage TPGInsertar;
        private System.Windows.Forms.Button BtnInsertar;
        private System.Windows.Forms.TextBox TxtCantidad;
        private System.Windows.Forms.TextBox TxtValorunitario;
        private System.Windows.Forms.TextBox TxtDescripcion;
        private System.Windows.Forms.TextBox TxtCodigo;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label LblVerificar;
        private System.Windows.Forms.ErrorProvider errorProvider1;
        private System.Windows.Forms.ErrorProvider errorProvider2;
        private System.Windows.Forms.Label LblCantiDADvERI;
        private System.Windows.Forms.Label LblValorVeri;
        private System.Windows.Forms.Label LblDescriVerificar;
    }
}