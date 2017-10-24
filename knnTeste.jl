using Distributions, DataFrames, PyPlot
function rsvdPredict(userId,itemId,U,V)
    u = U[userId,:]
    v = V[itemId,:]
    r = vecdot(u',v)
    if r > 5.0
      return 5.0
    elseif r < 1.0
      return 1.0
    end
    return r

end
#train a k column
function rsvdTrainK(usersItemsRating,U,V,k)
  lrate = 0.001
  λ = 0.02
  rmse = 0.0
  for index = 1:size(usersItemsRating,1)
      userId = usersItemsRating[index,1]
      itemId = usersItemsRating[index,2]
      trueRating = usersItemsRating[index,3]
      err = trueRating - rsvdPredict(userId,itemId,U,V)
      rmse += err^2
      u = U[userId,k]
      v = V[itemId,k]
      U[userId,k] += lrate*(err*v - λ*u)
      V[itemId,k] += lrate*(err*u - λ*v)
  end

  return U,V,sqrt(rmse/size(usersItemsRating,1))
end

function rsvdTrainBase(usersItemsRating,k_)
  U = zeros(Float64,totalUsers,k)
  V = zeros(Float64,totalItems,k)
  #inicializa U e V com valores aleatorios seguindo uma distribuicao normal
  for i = 1:totalUsers,j = 1:k
    U[i,j] = 0.1
  end
  for i = 1:totalItems,j = 1:k
    V[i,j] = 0.1
  end
  maxIter = 1000
  minErr = 0.0001
  previousErr = 1000000.0
  for aux in 1:k_
    println(aux)
    for count in 1:maxIter
      U,V,rmse = rsvdTrainK(usersItemsRating,U,V,aux)
      if abs(previousErr - rmse) < minErr
         break
      end
      previousErr = rmse
    end
  end
return U,V
end

function rsvdRmse(U,V,usersItemsRating)
  rmse = 0.0
  predicts = zeros(size(usersItemsRating,1))
  for index in 1:size(usersItemsRating,1)
    userId = usersItemsRating[index,1]
    itemId = usersItemsRating[index,2]
    trueRating = usersItemsRating[index,3]
    predicts[index]= rsvdPredict(userId,itemId,U,V)
    err = trueRating - rsvdPredict(userId,itemId,U,V)
    rmse += err^2
  end
  return sqrt(rmse/size(usersItemsRating,1)),predicts
end

treinamento1 = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)
teste1 = users = DataFrames.readtable("ml-100k/u1.test", separator=' ', header=false, names=u_col_names)

u,v = rsvdTrainBase(treinamento1,96)

function euclidean_distance(a, b)
 distance = 0.0
 for index in 1:size(a, 1)
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end
