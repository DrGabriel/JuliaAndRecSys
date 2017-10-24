using Distributions, DataFrames, PyPlot, knn, rsvd
u_col_names=[:user_id, :item_id, :rating, :timestamp]
treinamento1 = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)
teste1 = users = DataFrames.readtable("ml-100k/u1.test", separator=' ', header=false, names=u_col_names)
usuarioxitemTreino = zeros(Float64,943,1682)
for index in 1:size(treinamento1,1)
  i = treinamento1[index,1]
  j = treinamento1[index,2]
  nota = treinamento1[index,3]
  usuarioxitemTreino[i,j] = nota #note q havera varios zeros na matriz
end
usuarioxitemTeste = zeros(Float64,943,1682)
for index in 1:size(teste1,1)
  i = teste1[index,1]
  j = teste1[index,2]
  nota = teste1[index,3]
  usuarioxitemTeste[i,j] = nota #note q havera varios zeros na matriz
end


function detectaRuidoMahone(rating,predict,rmax,rmin,threshold)
  consistency = abs(rating - predict)/(rmax-rmin)
  return (consistency>threshold)
end
function noises_Mahone(usersItemsRatingTreino,usersItemsRatingTest,k)
  rmax = 5
  rmin = 1
  th = 0.01
  usersItemsRatingCopy = copy(usersItemsRatingTreino)
  #possible_noise = Tuple{Int, Int}[]
  for u in 1:size(usersItemsRatingTreino,1)
    user = usersItemsRatingTest[u,:]
    for i in 1:size(usersItemsRatingTreino,2)
      if usersItemsRatingTreino[u,i] !=0
        kNearestNeighbors,similaridades = pegakVizinhosCosin(usersItemsRatingTreino,user,u,i,k)
        if sum(kNearestNeighbors) != 0
            predict = calculaPredict(kNearestNeighbors,usersItemsRatingTreino,i,k,similaridades)
            if detectaRuidoMahone(usersItemsRatingTreino[u,i],predict,rmax,rmin,th) == true
              usersItemsRatingCopy[u,i] = predict
            end
        end
      end
    end
  end
  return usersItemsRatingCopy
end
function calculaKeV(usersItemsRating)
  Ku = zeros(Float64,size(usersItemsRating,1))
  Vu = zeros(Float64,size(usersItemsRating,1))
  Pu = zeros(Float64,size(usersItemsRating,1))
  Ki = zeros(Float64,size(usersItemsRating,2))
  Vi = zeros(Float64,size(usersItemsRating,2))


  for index in 1:size(usersItemsRating,1)
    ratings = usersItemsRating[index,:]
    ratingsIndexes = find(ratings)
    ratingNon0 = Int64[]
    for i in 1:size(ratingsIndexes,1)
      push!(ratingNon0,ratings[ratingsIndexes[i]])
    end

    x = mean(ratingNon0)
    p = std(ratingNon0)
    Pu[index] = p
    Ku[index] = x - p
    Vu[index] = x + p
  end

  for index in 1:size(usersItemsRating,2)
    ratings = usersItemsRating[:,index]
    ratingsIndexes = find(ratings)
    ratingNon0 = Int64[]
    for i in 1:size(ratingsIndexes,1)
      push!(ratingNon0,ratings[ratingsIndexes[i]])
    end
    x = mean(ratingNon0)
    p = std(ratingNon0)
    Ki[index] = x - p
    Vi[index] = x + p
  end
  return Ku,Vu,Ki,Vi,Pu
end

function detectaRuidoToledo(usersItemsRating)
  Ku,Vu,Ki,Vi,Pu = calculaKeV(usersItemsRating)
  Wu = zeros(Int64,size(usersItemsRating,1))
  Au = zeros(Int64,size(usersItemsRating,1))
  Su = zeros(Int64,size(usersItemsRating,1))
  Wi = zeros(Int64,size(usersItemsRating,2))
  Ai = zeros(Int64,size(usersItemsRating,2))
  Si = zeros(Int64,size(usersItemsRating,2))
  possible_noise = Tuple{Int, Int}[]
  for u in 1:size(usersItemsRating,1)
    for i in 1:size(usersItemsRating,2)
      if usersItemsRating[u,i] != 0
        rating = usersItemsRating[u,i]
        if rating < Ku[u]
          Wu[u]+=1
        elseif rating >= Ku[u] && rating < Vu[u]
          Au[u] +=1
        else
          Su[u]+=1
        end

        if rating < Ki[i]
          Wi[i]+=1
        elseif rating >= Ki[i] && rating < Vi[i]
          Ai[i]+=1
        else
          Si[i]+=1
        end
      end
    end
  end

  for u in 1:size(usersItemsRating,1)
    for i in 1:size(usersItemsRating,2)
        rating = usersItemsRating[u,i]
        if rating !=0
          if (Wu[u] >= Au[u] + Su[u]) && (Wi[i] >= Ai[i] + Si[i]) && rating >= Ku[u]
            push!(possible_noise,(u,i))
          end
          if (Au[u] >= Wu[u] + Su[u]) && (Ai[i] >= Wi[i] + Si[i]) && (rating < Ku[u] || rating >= Vu[u])
            push!(possible_noise,(u,i))
          end
          if (Su[u] >= Wu[u] + Au[u]) && (Si[i]>=Wi[i] + Ai[i]) && rating < Vu[u]
            push!(possible_noise,(u,i))
          end
        end

    end
  end
  return possible_noise,Pu
end

function corrigeRuidoToledo(usersItemsRatingTreino,usersItemsRatingTest,k)
  possible_noise,δ = detectaRuidoToledo(usersItemsRatingTreino)
  usersItemsRatingCorrigido = copy(usersItemsRatingTreino)
  for index in 1:size(possible_noise,1)
    u,i = possible_noise[index]
    user = usersItemsRatingTest[u,:]
    kNearestNeighbors,similaridades = pegakVizinhosCosin(usersItemsRatingCorrigido,user,u,i,k)
    if sum(kNearestNeighbors) != 0
      predict = calculaPredict(kNearestNeighbors,usersItemsRatingCorrigido,i,k,similaridades)
      if (abs(predict - usersItemsRatingCorrigido[u,i])>δ[u])
        usersItemsRatingCorrigido[u,i] = predict
      end
    end

  end
  return usersItemsRatingCorrigido
end
noise = noises_Mahone(usuarioxitemTreino,usuarioxitemTeste,4)

#knnBaseToda(usuarioxitemTreino,usuarioxitemTeste,teste1,4)
knnBaseToda(noise,usuarioxitemTeste,teste1,4)
