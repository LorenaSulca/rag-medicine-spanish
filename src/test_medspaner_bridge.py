from medspaner_bridge import run_medspaner

texto = "El paracetamol 1 g puede causar efectos adversos en pacientes con insuficiencia hepatica."

resultado = run_medspaner(texto)

print(resultado)
