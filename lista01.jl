using DataFrames, PyPlot
#Questão 1
u_col_names=[:user_id, :item_id, :rating, :timestamp]
users = DataFrames.readtable("ml-100k/u.data", separator=' ', header=false, names=u_col_names)
users_id = users[:,1]
avaliacoes_user = zeros(Int64,943)
sort!(users_id)
for x in 1:size(users_id,1)
  avaliacoes_user[users_id[x]] = avaliacoes_user[users_id[x]] + 1
end
sort!(avaliacoes_user)
plot(1:943, avaliacoes_user,color="red", linewidth=2.0, linestyle="-")

#questão 2
ratings = users[:,3]
fig = figure("pyplot_histogram",figsize=(10,10)) # Not strictly required
ax = axes() # Not strictly required
h = plt[:hist](ratings,5) # Histogram

grid("on")
xlabel("X")
ylabel("Y")
title("Histogram")
#Questao 3
treinamento1 = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)
teste1 = users = DataFrames.readtable("ml-100k/u1.test", separator=' ', header=false, names=u_col_names)
media_usuario = zeros(943)
avaliacoes_usuario = zeros(943)

for x in 1:size(treinamento1,1)
  media_usuario[treinamento1[x,1]] += treinamento1[x,3]
  avaliacoes_usuario[treinamento1[x,1]] += 1
end

media_usuario ./= avaliacoes_usuario #Media de cada usuario
erro = 0
for x in 1:size(teste1,1)
  erro += abs((media_usuario[teste1[x,1]] - teste1[x,3]))
end

println(erro/size(treinamento1,1))

#Questao 4
treinamento1 = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)
teste1 = users = DataFrames.readtable("ml-100k/u1.test", separator=' ', header=false, names=u_col_names)
media_item = zeros(1682)
avaliacoes_item = zeros(1682)

for x in 1:size(treinamento1,1)
  media_item[treinamento1[x,2]] += treinamento1[x,3]
  avaliacoes_item[treinamento1[x,2]] += 1
end
media_global = sum(media_item)/size(media_item,1)
for x in 1:1682
  if avaliacoes_item[x] == 0
    media_item[x] = media_global
  else
    media_item[x] = media_item[x]/avaliacoes_item[x]#Media de cada item
  end
end
erro = 0
for x in 1:size(teste1,1)
  erro += abs((media_item[teste1[x,2]] - teste1[x,3]))
end

println(erro/size(treinamento1,1))

#questao 5
treinamento1 = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)
teste1 = users = DataFrames.readtable("ml-100k/u1.test", separator=' ', header=false, names=u_col_names)
media_item = zeros(1682)
avaliacoes_item = zeros(1682)

for x in 1:size(treinamento1,1)
  media_item[treinamento1[x,2]] += treinamento1[x,3]
  avaliacoes_item[treinamento1[x,2]] += 1
end
media_global = sum(media_item)/size(media_item,1)
for x in 1:1682
  if avaliacoes_item[x] == 0
    media_item[x] = media_global
  else
    media_item[x] = media_item[x]/avaliacoes_item[x]#Media de cada item
  end
end

media_usuario = zeros(943)
avaliacoes_usuario = zeros(943)

for x in 1:size(treinamento1,1)
  media_usuario[treinamento1[x,1]] += treinamento1[x,3]
  avaliacoes_usuario[treinamento1[x,1]] += 1
end
media_usuario ./= avaliacoes_usuario #Media de cada usuario

usuarioXitem = zeros(Float64,943,1682)
for i in 1:943
  for j in 1:1682
    usuarioXitem[i,j] = (media_usuario[i] + media_item[j])/2
  end
end

erro = 0
for x in 1:size(teste1,1)
  erro += abs((usuarioXitem[teste1[x,1],teste1[x,2]] - teste1[x,3]))
end

println(erro/size(treinamento1,1))
