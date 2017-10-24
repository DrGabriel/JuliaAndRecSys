f= open("ml-100k/u1.test")
f1= open("ml-100k/u1.base")

#inicia as variaveis utilizadas com 0
treinamento=zeros(943,1682)
test=zeros(462,1682)
pearson=zeros(size(test,1),943)
usuarios=zeros(size(test,1),943)
previsao=zeros(size(test,1),1682)
somatorio = 0.0
norma = 0.0
err=0
rmse=0
cont=0

#faz a leitura dos arquivos
function rf(f,c)
    parse(Int,(readuntil(f,c)))
end

while(!eof(f))
  	usr =rf(f,'\t')
 	it =rf(f,'\t')
	test[usr,it]= rf(f,'\t')
	rf(f,'\n')
end

while(!eof(f1))
  	usr =rf(f1,'\t')
 	it =rf(f1,'\t')
	treinamento[usr,it]= rf(f1,'\t')

	rf(f1,'\n')
end

#calcula o pearson
for i in 1:size(test,1)
	for j in 1:943
    	pearson[i,j]=cor(treinamento[j,:], test[i,:])
   	end
end

#separa os vizinhos mais proximos
for i in 1:size(test,1)
	for j in 1:943
		if(pearson[i,j]>0.7)
			usuarios[i,j]=1
		elseif(pearson[i,j]<-0.7)
			usuarios[i,j]=1
		end
	end
end

#faz a previsão a partir da formula do knn
for k in 1:1682
  	for i in 1:size(test,1)
    	for j in 1:943
     		if(usuarios[i,j]==1)
          		somatorio+=treinamento[j,k]*abs(pearson[i,j])
          		norma += abs(pearson[i,j])
          		previsao[i,k]=somatorio/norma
        	end
    	end
    end
end

#calcula o rmse
for i in 1:size(test,1)
	for j in 1:1682
		if(previsao[i,j]-test[i,j]!=0 && test[i,j]!=0)
			cont=cont+1
			err=previsao[i,j]-test[i,j]
			rmse+=err*err
		end
	end
end

err=sqrt(rmse/cont)
