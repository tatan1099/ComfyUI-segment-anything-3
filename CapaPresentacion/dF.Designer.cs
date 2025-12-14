namespace CapaPresentacion
{
    partial class dF
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
            this.TxtNFactura = new System.Windows.Forms.TextBox();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.textBox4 = new System.Windows.Forms.TextBox();
            this.DtFecha = new System.Windows.Forms.DateTimePicker();
            this.CmbCLIENTE = new System.Windows.Forms.ComboBox();
            this.SuspendLayout();
            // 
            // TxtNFactura
            // 
            this.TxtNFactura.Location = new System.Drawing.Point(103, 94);
            this.TxtNFactura.Name = "TxtNFactura";
            this.TxtNFactura.Size = new System.Drawing.Size(100, 20);
            this.TxtNFactura.TabIndex = 0;
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(92, 262);
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(100, 20);
            this.textBox2.TabIndex = 1;
            // 
            // textBox4
            // 
            this.textBox4.Location = new System.Drawing.Point(92, 218);
            this.textBox4.Name = "textBox4";
            this.textBox4.Size = new System.Drawing.Size(100, 20);
            this.textBox4.TabIndex = 3;
            // 
            // DtFecha
            // 
            this.DtFecha.Location = new System.Drawing.Point(82, 131);
            this.DtFecha.Name = "DtFecha";
            this.DtFecha.Size = new System.Drawing.Size(200, 20);
            this.DtFecha.TabIndex = 4;
            // 
            // CmbCLIENTE
            // 
            this.CmbCLIENTE.FormattingEnabled = true;
            this.CmbCLIENTE.Location = new System.Drawing.Point(82, 172);
            this.CmbCLIENTE.Name = "CmbCLIENTE";
            this.CmbCLIENTE.Size = new System.Drawing.Size(121, 21);
            this.CmbCLIENTE.TabIndex = 5;
            // 
            // dF
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.CmbCLIENTE);
            this.Controls.Add(this.DtFecha);
            this.Controls.Add(this.textBox4);
            this.Controls.Add(this.textBox2);
            this.Controls.Add(this.TxtNFactura);
            this.Name = "dF";
            this.Text = "dF";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox TxtNFactura;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.TextBox textBox4;
        private System.Windows.Forms.DateTimePicker DtFecha;
        private System.Windows.Forms.ComboBox CmbCLIENTE;
    }
}