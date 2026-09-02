import numpy as np

# Saulo da Silva Benício – 2314224 
# Daniel de Carvalho Moreira - 2310419 

#Coletar os dados da planilha:
dados = np.loadtxt(fname="C:/Users/Saulobeni/Desktop/Dataset_Projeto_IA.csv",delimiter=",",skiprows=1)

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