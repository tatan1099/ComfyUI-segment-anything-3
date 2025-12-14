using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CapaNegocios
{
    public class validaciones
    {
        public static string Encriptacion(string contraseña)
        {
            using (SHA256 sha256Hash = SHA256.Create())
            {
                // Convertimos la contraseña en bytes
                byte[] bytes = sha256Hash.ComputeHash(Encoding.UTF8.GetBytes(contraseña));

                // Convertimos los bytes en una cadena hexadecimal
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < bytes.Length; i++)
                {
                    builder.Append(bytes[i].ToString("x2"));
                }
                return builder.ToString();
            }
        }
        public bool ValidarCorreo(string correo)
        {
            string Emailformato;
            Emailformato = "\\w+([-+.']\\w+)*@\\w+([-.]\\w+)*\\.\\w+([-.]\\w+)*";
            if (Regex.IsMatch(correo, Emailformato))
            {
                if (Regex.Replace(correo, Emailformato, String.Empty).Length == 0)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        public bool CheckProductCodeExists(int CODproducto)
        {
            SqlConnection conexion = new SqlConnection("Server=BUCDFPCSEFSD038;DataBase=NaturVida;Integrated Security=true");
            bool exists = false;
            SqlCommand command = new SqlCommand("SELECT COUNT(*) FROM Productos WHERE Codigo = @Codigo", conexion);
            command.Parameters.AddWithValue("@Codigo", CODproducto);
            conexion.Open();
            int count = (int)command.ExecuteScalar();
            if (count > 0)
            {
                exists = true;
            }
            conexion.Close();
            return exists;
        }
        public bool CheckDocExists(int Documento)
        {
            SqlConnection conexion = new SqlConnection("Server=BUCDFPCSEFSD038;DataBase=NaturVida;Integrated Security=true");
            bool exists = false;
            SqlCommand command = new SqlCommand("SELECT COUNT(*) FROM Cliente WHERE Documento = @Documento", conexion);
            command.Parameters.AddWithValue("@Documento", Documento);
            conexion.Open();
            int count = (int)command.ExecuteScalar();
            if (count > 0)
            {
                exists = true;
            }
            conexion.Close();
            return exists;
        }
        public bool CheckProductDescrioExists(string Descripcion)
        {
            SqlConnection conexion = new SqlConnection("Server=BUCDFPCSEFSD038;DataBase=NaturVida;Integrated Security=true");
            bool exists = false;
            SqlCommand command = new SqlCommand("SELECT COUNT(*) FROM Productos WHERE Descripción = @Descripcion", conexion);
            command.Parameters.AddWithValue("@Descripcion", Descripcion);
            conexion.Open();
            int count = (int)command.ExecuteScalar();
            if (count > 0)
            {
                exists = true;
            }
            conexion.Close();
            return exists;
        }
        public bool CheckUserExists(string User)
        {
            SqlConnection conexion = new SqlConnection("Server=BUCDFPCSEFSD038;DataBase=NaturVida;Integrated Security=true");
            bool exists = false;
            SqlCommand command = new SqlCommand("SELECT COUNT(*) FROM Vendedores WHERE Usuario = @Usuario", conexion);
            command.Parameters.AddWithValue("@Usuario", User);
            conexion.Open();
            int count = (int)command.ExecuteScalar();
            if (count > 0)
            {
                exists = true;
            }
            conexion.Close();
            return exists;
        }
        public bool CheckCodeVender(int CODproducto)
        {
            SqlConnection conexion = new SqlConnection("Server=BUCDFPCSEFSD038;DataBase=NaturVida;Integrated Security=true");
            bool exists = false;
            SqlCommand command = new SqlCommand("SELECT COUNT(*) FROM Vendedores WHERE Codigo = @Codigo", conexion);
            command.Parameters.AddWithValue("@Codigo", CODproducto);
            conexion.Open();
            int count = (int)command.ExecuteScalar();
            if (count > 0)
            {
                exists = true;
            }
            conexion.Close();
            return exists;
        }

    }
}
