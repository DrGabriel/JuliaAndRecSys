#Questao 1
matrix = rand(0:10,10,10)

for i in size(matrix,1)
  matrix[i,indmin(matrix[i,:])] = 0
  matrix[i,indmax(matrix[i,:])] = 1
end

#Questao 2
M = [5 10 -5 22; 1 33 15 3; 8 29 12 1; 3 11 39 20]
for i in 1:3
  M[indmax(M)] = 0
end

#Questao 3
using Distributions
M = rand(5,5)
function normalDistribution(matrix)
  d = Normal()
  for j=1:size(matrix,2)
    for i=1:size(matrix,1)
      matrix[i,j] = rand(d)
    end
  end
end
normalDistribution(M)
