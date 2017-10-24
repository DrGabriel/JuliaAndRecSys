module knn
  using Distributions, DataFrames, PyPlot
  export knnBaseToda,pegakVizinhosCosin,calculaPredict
  u_col_names=[:user_id, :item_id, :rating, :timestamp]
  treinamento1 = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)
  teste1 = users = DataFrames.readtable("ml-100k/u1.test", separator=' ', header=false, names=u_col_names)
  #monto a matriz usuario x item
  usuarioxitemTreino = zeros(Int64,943,1682)
  for index in 1:size(treinamento1,1)
    i = treinamento1[index,1]
    j = treinamento1[index,2]
    nota = treinamento1[index,3]
    usuarioxitemTreino[i,j] = nota #note q havera varios zeros na matriz
  end
  #monto a matriz usuario x item do teste
  usuarioxitemTeste = zeros(Int64,943,1682)
  for index in 1:size(teste1,1)
    i = teste1[index,1]
    j = teste1[index,2]
    nota = teste1[index,3]
    usuarioxitemTeste[i,j] = nota #note q havera varios zeros na matriz
  end
  function euclidean_distance(a, b)
   distance = 0.0
   for index in 1:size(a, 1)
    distance += (a[index]-b[index]) * (a[index]-b[index])
   end
   return 1+sqrt(distance)
  end
  function euclidean_distance_inverse(a, b)
   distance = 0.0
   for index in 1:size(a, 1)
    distance += (a[index]-b[index]) * (a[index]-b[index])
   end
   return 1/(1+sqrt(distance))
  end
  #user-based
  function pegakVizinhosEuclideanInverse(usuarioxitemTreino,usuarioTeste,indiceUser,indiceItem,k)
    #quero pegar usuarios proximos q avaliaram o item i
    similaridades = zeros(Float64,943)
    totalVizinhos = 0
    for index in 1:943
      usuarioTreino = usuarioxitemTreino[index,:]
      #pode ser que o usuario em questão nao exista no treinamento, isto é todos os ratings são 0
      if sum(usuarioxitemTreino[index,:]) !=0 && index != indiceUser && usuarioTreino[indiceItem] != 0  #evita q eu compare o usuario com ele mesmo
          similaridades[index] = euclidean_distance_inverse(usuarioTeste,usuarioTreino)
          totalVizinhos +=1
      else
        similaridades[index] = -10000000
      end
    end
    if totalVizinhos < k
      return 0,0
    else
     sortedNeighbors = sortperm(similaridades,rev=true)
     kNearestNeighbors = sortedNeighbors[1:k]
     return kNearestNeighbors,similaridades
   end
  end
  function pegakVizinhos(usuarioxitemTreino,usuarioTeste,indiceUser,indiceItem,k)
    #quero pegar usuarios proximos q avaliaram o item i
    similaridades = zeros(Float64,943)
    totalVizinhos = 0
    for index in 1:943
      usuarioTreino = usuarioxitemTreino[index,:]
      #pode ser que o usuario em questão nao exista no treinamento, isto é todos os ratings são 0
      if sum(usuarioxitemTreino[index,:]) !=0 && index != indiceUser && usuarioTreino[indiceItem] != 0  #evita q eu compare o usuario com ele mesmo
          similaridades[index] = euclidean_distance(usuarioTeste,usuarioTreino)
          totalVizinhos +=1
      else
        similaridades[index] = 10000000
      end
    end
    if totalVizinhos < k
      return 0,0
    else
     sortedNeighbors = sortperm(similaridades)
     kNearestNeighbors = sortedNeighbors[1:k]
     return kNearestNeighbors,similaridades
   end
  end

  function pegakVizinhosCosin(usuarioxitemTreino,usuarioTeste,indiceUser,indiceItem,k)
    #quero pegar usuarios proximos q avaliaram o item i
    similaridades = zeros(Float64,943)
    totalVizinhos = 0
    for index in 1:943
      usuarioTreino = usuarioxitemTreino[index,:]
      #pode ser que o usuario em questão nao exista no treinamento, isto é todos os ratings são 0
      if sum(usuarioxitemTreino[index,:]) !=0 && index != indiceUser && usuarioTreino[indiceItem] != 0  #evita q eu compare o usuario com ele mesmo
          similaridades[index] = dot(usuarioTeste, usuarioTreino) / (norm(usuarioTeste) * norm(usuarioTreino))
          totalVizinhos +=1
      else
        similaridades[index] = -10000000
      end
    end
    if totalVizinhos < k
      return 0,0
    else
     sortedNeighbors = sortperm(similaridades,rev=true)
     kNearestNeighbors = sortedNeighbors[1:k]
     return kNearestNeighbors,similaridades
   end
  end

  function calculaPredict(kVizinhos,usuarioxitemTreino,indiceItem,k,similaridades)
    somatorio = 0.0
    norma = 0.0
    for indice in 1:k
      somatorio += usuarioxitemTreino[kVizinhos[indice],indiceItem] * similaridades[kVizinhos[indice]]
      norma += abs(similaridades[kVizinhos[indice]])
    end
    return (somatorio/norma)
  end

  k = 4
  function knnBaseToda(usuarioxitemTreino,usuarioxitemTeste,testeSet,k)
    rmse = 0.0
    tam = 0
    for indice in 1:size(testeSet,1)
      indiceUser = testeSet[indice,1]
      indiceItem = testeSet[indice,2]
      notaReal = testeSet[indice,3]
      user = usuarioxitemTeste[indiceUser,:]
      kvizinhos,similaridades = pegakVizinhosEuclideanInverse(usuarioxitemTreino,user,indiceUser,indiceItem,k)
      if sum(kvizinhos) != 0
        tam +=1
        previsao = calculaPredict(kvizinhos,usuarioxitemTreino,indiceItem,k,similaridades)
        err = notaReal - previsao
        rmse += err^2
      end
    end
    println(sqrt(rmse/tam))
  end
end
