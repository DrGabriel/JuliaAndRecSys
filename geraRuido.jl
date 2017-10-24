using Distributions, DataFrames

totalUsers = 943
totalItems = 1682

# retorna true ou false de acordo
# com a probabilidade prob
function retornaSucesso(prob)
    x = rand(1:100)
    if(x <= prob)
        return true;
    else
        return false;
    end
end

function geraRuido(base)
    for i in 1:size(base,1)
        if(retornaSucesso(20))
            if(retornaSucesso(50))
                base[i,3] = 1
            else
                base[i,3] = 5
            end
        end
    end
end

u_col_names=[:user_id, :item_id, :rating, :timestamp]

treinamento1 = DataFrames.readtable("ml-100k/u1.base", separator=' ', header=false, names=u_col_names)
teste1 = users = DataFrames.readtable("ml-100k/u1.test", separator=' ', header=false, names=u_col_names)

geraRuido(treinamento1)
