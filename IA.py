import numpy as np
import os

# Saulo da Silva Benício – 2314224 
# Daniel de Carvalho Moreira - 2310419 

PASTA_DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
CAMINHO_ARSENIO = os.path.join(PASTA_DOWNLOADS, "arsenio_dataset.csv")
CAMINHO_DOSE = os.path.join(PASTA_DOWNLOADS, "dose_radiacao_expandido.csv")

# ============================================================
# PROBLEMA 1 - Arsênio nas unhas (mantido igual ao original)
# ============================================================

#Coletar os dados da planilha:
dados = np.loadtxt(fname=CAMINHO_ARSENIO, delimiter=",", skiprows=1)

#Separação dos dados por 'coluna'(tipo):
dadosIdade = dados[:,0]
dadosSexo = dados[:,1]
dadosBeber = dados[:,2]
dadosCozinhar = dados[:,3]
dadosArsenioNaAgua = dados[:,4]
dadosArsenioNaUnha = dados[:,5]
   
#Questao a :
y = dadosArsenioNaUnha

x1 = dadosIdade
x2 = dadosBeber
x3 = dadosCozinhar
x4 = dadosArsenioNaAgua
X = np.column_stack((np.ones(len(x1)),x1,x2,x3,x4))
Xt = X.T
Beta = np.linalg.pinv(Xt@X)@(Xt@y)
equacao_print = f"y = {Beta[0]:.3f} + {Beta[1]:.3f}*X1 + {Beta[2]:.3f}*X2 + {Beta[3]:.3f}*X3 + {Beta[4]:.3f}*X4"
print("=== PROBLEMA 1 - Arsênio nas unhas ===")
print(equacao_print)

#Questão b: Use o modelo para prever o arsênio nas unhas(y), 
# quando a idade for 30(x1), a categoria da agua para beber for 5(x2),
# a categoria da agua para cozinhar for 5(x3) e o arsênio na agua for
# 0,135(x4) ppm.
yPrevistoParaItemB = Beta[0] + Beta[1]*30 + Beta[2]*5 + Beta[3]*5 + Beta[4]*0.135
print(f"Para os devidos valores de idade(x1=30 anos), \n categoria de água para beber(x2=5), \n categoria de água para cozinhar(x3=5) \n e arsênio na água(x4=0,135) :\n  A quantidade de arsênio das unhas é = {yPrevistoParaItemB}")

#Questão d:Qual  ‌e o valor de R2 score para esse modelo?

yEstimado = X@Beta

ssTotal = np.sum((y-y.mean())**2)
ssResp = np.sum((y-yEstimado)**2)
rScore = 1 - ssResp/ssTotal
print(f"R² = {rScore}")

#Questão e: Muitos usuários de regressão preferem usar uma estatística de valor ajustado de R2.
# Por quê? Ela foi melhor que R2 comum? Se sim, por quê?

rAjustado = 1 - ((1- rScore)*(len(x1)-1))/(len(x1)-4-1)
print(f"Rajustado =  {rAjustado}")

#Questão f1: Compare este modelo com um modelo alternativo que use apenas a concentração de
#arsênio na  ‌agua como preditor. Qual modelo  ‌e melhor? Por que?
yF = y
xF = x4
xMed = np.mean(xF)
yMed = np.mean(yF)
sup = np.sum((xF-xMed)*(yF-yMed))
inf = np.sum((xF-xMed)**2)
a = sup/inf
b = yMed - a*(xMed)
yTeste = a*xF + b
cimar = (np.sum((yF - yTeste)**2)) 
baixor = np.sum((yF-yMed)**2)
r2=1 - (cimar/baixor)
print(f"R²(para a regressão linear simples[1 varíavel]) =  {r2}")

#Questão f2: Realize uma análise de resíduos para verificar as suposições do modelo de regressão.
#Para isso, siga os seguintes passos:

#primeira parte: Calcule os valores ajustados para todas as observações de y:
yPred_eq = []
for i in range(len(x1)):
    yPred =  Beta[0] + Beta[1]*x1[i] + Beta[2]*x2[i] + Beta[3]*x3[i] + Beta[4]*x4[i]
    yPred_eq = np.append(yPred_eq,yPred)
yPred_eq_print = np.array(yPred_eq)


#Segunda parte: Calcule os resíduos correspondentes:
e = y - yPred_eq_print
eF = yF - yTeste

#Terceira Parte: 
tabela = []
for i in range(len(y)):   
    linha = [i+1, y[i], yPred_eq_print[i], e[i]]
    tabela.append(linha)
  
tabelaF =[]  
for i in range(len(yF)):
    linhaF = [i+1,yF[i],yTeste[i],eF[i]]
    tabelaF.append(linhaF)
    
tabela = np.array(tabela)
tabelaF = np.array(tabelaF)

# Imprimir a tabela com cabeçalho
print("\nTabela do modelo de Regressão Linear Múltipla")
print(" i\t y\t y_pred\t erro")
for row in tabela:
    print(f"{int(row[0])}\t {row[1]:.4f}\t {row[2]:.4f}\t {row[3]:.4f}")
 
print("\nTabela do modelo de Regressão Linear Simples")   
print(" i\t y\t y_pred\t erro")
for row in tabelaF:
    print(f"{int(row[0])}\t {row[1]:.4f}\t {row[2]:.4f}\t {row[3]:.4f}")
    
#Vamos observar a média dos erros:
eMed = np.abs(np.mean(e))
eFMed = np.abs(np.mean(eF))
print(f"\n|Erro múltiplo| x |Erro simples| \n    {eMed}   x    {eFMed:}")
#Com isso podemos ver que o menor erro é proporcionado pela Regressão Linear Múltipla, devido a ela possuir mais varíaveis , oq permite uma maior precisão na sua predição

# # Questão g:
# Modelo com intercepto zero (sem coluna de 1s)
X_zero_intercept = np.column_stack((x1, x2, x3, x4))
Beta_zero_intercept = np.linalg.pinv(X_zero_intercept.T @ X_zero_intercept) @ (X_zero_intercept.T @ y)
y_estimado_zero_intercept = X_zero_intercept @ Beta_zero_intercept
ss_res_zero = np.sum((y - y_estimado_zero_intercept) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2_zero_intercept = 1 - ss_res_zero / ss_tot
rmse_zero_intercept = np.sqrt(np.mean((y - y_estimado_zero_intercept) ** 2))
rmse_com_intercepto = np.sqrt(np.mean((y - yEstimado) ** 2))
r2_com_intercepto = rScore

print(f"\n\nModelo com intercepto zero: R² = {r2_zero_intercept:.4f}, RMSE = {rmse_zero_intercept:.4f}")
print(f"Modelo com intercepto livre: R² = {r2_com_intercepto:.4f}, RMSE = {rmse_com_intercepto:.4f}")


#Questão h:
mse_completo = np.mean((y - yEstimado) ** 2)
mae_completo = np.mean(np.abs(y - yEstimado))

# Modelo arsênio na água
mse_simples = np.mean((yF - yTeste) ** 2)
mae_simples = np.mean(np.abs(yF - yTeste))
print("Modelo completo (múltiplas variáveis):")
print(f"MSE = {mse_completo:.6f}")
print(f"MAE = {mae_completo:.6f}")
print("\nModelo alternativo (somente arsênio na água):")
print(f"MSE = {mse_simples:.6f}")
print(f"MAE = {mae_simples:.6f}")


# ============================================================
# PROBLEMA 2 - Dose de radiação em circuitos integrados (NOVO)
# Dataset: dose_radiacao_expandida.csv
# Colunas esperadas: Dose de Radiação (resposta), Corrente (mAmp), Tempo de Exposição
# ============================================================

print("\n\n=========================================")
print("=== PROBLEMA 2 - Dose de Radiação ===")
print("=========================================")

# O CSV real (dose_radiacao_expandido.csv) tem uma coluna extra de índice
# no início (sem cabeçalho nomeado), por isso usamos usecols para pular
# a coluna 0 e ler apenas Dose_de_Radiacao, mAmp e Tempo_de_Exposicao.
dados2 = np.loadtxt(fname=CAMINHO_DOSE, delimiter=",", skiprows=1, usecols=(1,2,3))

# Separação das colunas
dadosDose = dados2[:,0]        # Dose de Radiação (resposta)
dadosCorrente = dados2[:,1]    # Corrente (mAmp)
dadosTempo = dados2[:,2]       # Tempo de Exposição (min)

#Questão a: Ajuste um modelo de regressão linear múltipla com dose de radiação
# como variável resposta (Corrente e Tempo de Exposição como preditores)
y2 = dadosDose
z1 = dadosCorrente
z2 = dadosTempo

X2 = np.column_stack((np.ones(len(z1)), z1, z2))
X2t = X2.T
Beta2 = np.linalg.pinv(X2t@X2)@(X2t@y2)
equacao2_print = f"y = {Beta2[0]:.3f} + {Beta2[1]:.3f}*Corrente + {Beta2[2]:.3f}*Tempo"
print(equacao2_print)

#Questão b: Use o modelo para prever a dose de radiação quando a corrente
# for de 15 miliamperes e o tempo de exposição for de 5 minutos.
yPrevistoParaItemB2 = Beta2[0] + Beta2[1]*15 + Beta2[2]*5
print(f"Para corrente=15 mA e tempo de exposição=5 min:\n  A dose de radiação prevista é = {yPrevistoParaItemB2}")

#Questão c: Qual é o valor de R² score para esse modelo?
y2Estimado = X2@Beta2
ssTotal2 = np.sum((y2-y2.mean())**2)
ssResp2 = np.sum((y2-y2Estimado)**2)
rScore2 = 1 - ssResp2/ssTotal2
print(f"R² = {rScore2}")

#Questão d: Muitos usuários de regressão preferem usar uma estatística de valor ajustado de R2.
# Por quê? Ela foi melhor que R2 comum? Se sim, por quê?
rAjustado2 = 1 - ((1 - rScore2)*(len(z1)-1))/(len(z1)-2-1)
print(f"Rajustado = {rAjustado2}")

#Questão e: Compare este modelo com um modelo alternativo que use apenas
# a Corrente como preditor. Qual modelo é melhor? Por que?
yF2 = y2
xF2 = z1
xMed2 = np.mean(xF2)
yMed2 = np.mean(yF2)
sup2 = np.sum((xF2-xMed2)*(yF2-yMed2))
inf2 = np.sum((xF2-xMed2)**2)
a2 = sup2/inf2
b2 = yMed2 - a2*(xMed2)
yTeste2 = a2*xF2 + b2
cimar2 = np.sum((yF2 - yTeste2)**2)
baixor2 = np.sum((yF2-yMed2)**2)
r2_simples2 = 1 - (cimar2/baixor2)
print(f"R²(para a regressão linear simples[apenas Corrente]) = {r2_simples2}")

print(f"\nComparação: R² múltiplo = {rScore2:.4f}  x  R² simples (Corrente) = {r2_simples2:.4f}")
print("O modelo com melhor R² (e menor erro) é considerado o mais adequado.")

#Questão f: Cenário com Intercepto Forçado a Zero
X2_zero_intercept = np.column_stack((z1, z2))
Beta2_zero_intercept = np.linalg.pinv(X2_zero_intercept.T @ X2_zero_intercept) @ (X2_zero_intercept.T @ y2)
y2_estimado_zero_intercept = X2_zero_intercept @ Beta2_zero_intercept
ss_res_zero2 = np.sum((y2 - y2_estimado_zero_intercept) ** 2)
ss_tot2 = np.sum((y2 - y2.mean()) ** 2)
r2_zero_intercept2 = 1 - ss_res_zero2 / ss_tot2
rmse_zero_intercept2 = np.sqrt(np.mean((y2 - y2_estimado_zero_intercept) ** 2))
rmse_com_intercepto2 = np.sqrt(np.mean((y2 - y2Estimado) ** 2))
r2_com_intercepto2 = rScore2

print(f"\nModelo com intercepto zero: R² = {r2_zero_intercept2:.4f}, RMSE = {rmse_zero_intercept2:.4f}")
print(f"Modelo com intercepto livre: R² = {r2_com_intercepto2:.4f}, RMSE = {rmse_com_intercepto2:.4f}")

#Questão h: Outras métricas de erro (MSE, RMSE, MAE) - modelo completo e alternativo (apenas Corrente)
mse_completo2 = np.mean((y2 - y2Estimado) ** 2)
mae_completo2 = np.mean(np.abs(y2 - y2Estimado))

mse_simples2 = np.mean((yF2 - yTeste2) ** 2)
mae_simples2 = np.mean(np.abs(yF2 - yTeste2))

print("\nModelo completo (Corrente + Tempo):")
print(f"MSE = {mse_completo2:.6f}")
print(f"RMSE = {rmse_com_intercepto2:.6f}")
print(f"MAE = {mae_completo2:.6f}")
print("\nModelo alternativo (somente Corrente):")
print(f"MSE = {mse_simples2:.6f}")
print(f"RMSE = {np.sqrt(mse_simples2):.6f}")
print(f"MAE = {mae_simples2:.6f}")