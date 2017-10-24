using DataFrames
# Retorna Array de tupples com indices (usr,itm)
# das notas que foram alteradas
function novaNota(notaAnt)
     x = rand(1:5)
     while(x == notaAnt)
         x = rand(1:5)
     end
     return x
end
function geraRuido(tamBase,percPerturb,itmUsr)
    a = Array{Tuple{Int}}(tamBase)
    tamanhoRuido = tamBase*percPerturb/100
    for i in 1:tamanhoRuido
        usr = rand(1:943)
        itm = rand(1:1682)
        a[i] = (usr,itm)
        itmUsr = novaNota(itmUsr[usr,itm])
    end
    return a
end



totalItems = 1682
totalUsers = 943

u_col_names=[:user_id, :item_id, :rating, :timestamp]

treinamento = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)


usuarioxitemTeste = zeros(Int,943,1682)
for index in 1:size(treinamento,1)
  i = treinamento[index,1]
  j = treinamento[index,2]
  nota = treinamento[index,3]
  usuarioxitemTeste[i,j] = nota #note q havera varios zeros na matriz
end

ruido = geraRuido(size(treinamento,1),20,usuarioxitemTeste)
