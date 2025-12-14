namespace CapaPresentacion
{
    partial class Inventario
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
            this.DgvInventario = new System.Windows.Forms.DataGridView();
            this.comboBox1 = new System.Windows.Forms.ComboBox();
            this.BtnMostrarTodo = new System.Windows.Forms.Button();
            this.BtnMostrarConsulta = new System.Windows.Forms.Button();
            this.button1 = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.DgvInventario)).BeginInit();
            this.SuspendLayout();
            // 
            // DgvInventario
            // 
            this.DgvInventario.BackgroundColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.DgvInventario.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.DgvInventario.Location = new System.Drawing.Point(8, 212);
            this.DgvInventario.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.DgvInventario.Name = "DgvInventario";
            this.DgvInventario.Size = new System.Drawing.Size(894, 187);
            this.DgvInventario.TabIndex = 0;
            // 
            // comboBox1
            // 
            this.comboBox1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.comboBox1.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.comboBox1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.comboBox1.FormattingEnabled = true;
            this.comboBox1.Location = new System.Drawing.Point(254, 64);
            this.comboBox1.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.comboBox1.Name = "comboBox1";
            this.comboBox1.Size = new System.Drawing.Size(491, 33);
            this.comboBox1.TabIndex = 1;
            // 
            // BtnMostrarTodo
            // 
            this.BtnMostrarTodo.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnMostrarTodo.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnMostrarTodo.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnMostrarTodo.Location = new System.Drawing.Point(564, 141);
            this.BtnMostrarTodo.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.BtnMostrarTodo.Name = "BtnMostrarTodo";
            this.BtnMostrarTodo.Size = new System.Drawing.Size(321, 63);
            this.BtnMostrarTodo.TabIndex = 2;
            this.BtnMostrarTodo.Text = "Mostrar inventarios";
            this.BtnMostrarTodo.UseVisualStyleBackColor = false;
            this.BtnMostrarTodo.Click += new System.EventHandler(this.BtnMostrarTodo_Click);
            // 
            // BtnMostrarConsulta
            // 
            this.BtnMostrarConsulta.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.BtnMostrarConsulta.Font = new System.Drawing.Font("MV Boli", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.BtnMostrarConsulta.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.BtnMostrarConsulta.Location = new System.Drawing.Point(160, 141);
            this.BtnMostrarConsulta.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.BtnMostrarConsulta.Name = "BtnMostrarConsulta";
            this.BtnMostrarConsulta.Size = new System.Drawing.Size(321, 63);
            this.BtnMostrarConsulta.TabIndex = 3;
            this.BtnMostrarConsulta.Text = "Consultar";
            this.BtnMostrarConsulta.UseVisualStyleBackColor = false;
            this.BtnMostrarConsulta.Click += new System.EventHandler(this.BtnMostrarConsulta_Click);
            // 
            // button1
            // 
            this.button1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(246)))), ((int)(((byte)(186)))), ((int)(((byte)(111)))));
            this.button1.Font = new System.Drawing.Font("MV Boli", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(235)))), ((int)(((byte)(235)))));
            this.button1.Location = new System.Drawing.Point(1275, 16);
            this.button1.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(244, 58);
            this.button1.TabIndex = 4;
            this.button1.Text = "Volver";
            this.button1.UseVisualStyleBackColor = false;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // Inventario
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 17F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(173)))), ((int)(((byte)(228)))), ((int)(((byte)(219)))));
            this.ClientSize = new System.Drawing.Size(942, 509);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.BtnMostrarConsulta);
            this.Controls.Add(this.BtnMostrarTodo);
            this.Controls.Add(this.comboBox1);
            this.Controls.Add(this.DgvInventario);
            this.Font = new System.Drawing.Font("MV Boli", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.Name = "Inventario";
            this.Text = "Inventario";
            this.Load += new System.EventHandler(this.Inventario_Load);
            ((System.ComponentModel.ISupportInitialize)(this.DgvInventario)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.DataGridView DgvInventario;
        private System.Windows.Forms.ComboBox comboBox1;
        private System.Windows.Forms.Button BtnMostrarTodo;
        private System.Windows.Forms.Button BtnMostrarConsulta;
        private System.Windows.Forms.Button button1;
    }
}