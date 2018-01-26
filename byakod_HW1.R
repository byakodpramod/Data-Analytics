# Pramod Aravind Byakod
# OU ID - 113436879

#Question 1
#Create a vector "x" and assign values
x = c(3, 12, 6, -5, 0, 8, 15, 1, -10, 7)
#display value of vector "x"
x 
#Create a vector "y" and assign values using seq command
y = seq(min(x),max(x),length=10) 
#Compute sum of x
sum(x) 
#Compute mean of x  
mean(x) 
#Compute standard deviation of x                                  
sd(x) 
#Load the lsr package
library(lsr) 
#Compute mean absolute deviation of x
aad(x) 
#Compute variance of x
var(x) 
#Compute sum of y
sum(y) 
#Compute mean of y
mean(y) 
#Compute standard deviation of y
sd(y) 
#Compute mean absolute deviation of y
aad(y) 
#Compute variance of y
var(y) 
#load the package "moments". Used to compute skewness and kurtosis
library(moments) 
#find skewness of x
skewness(x) 
#find kurtosis of x
kurtosis(x) 
#compute a statistical test for differences in means between the vectors x and y
t.test(x,y) 
#Sort the vector x
x = sort(x) 
#Paired test on x and y after x has been sorted
t.test(x,y,paired=TRUE) 
#Differences in mean are not significant
#Create a logical vector to identify the negetive values in x
x_neg = x<0 
#Display the previously created logical vector
x_neg 
#Remove all the negetive values from x
x = x[!x_neg] 
#Question 2
#Read the .csv file and store data into a dataframe
college = read.csv("college.csv") 
#Remove the first coloumn from college
college <- college [,-1] 
#Display summary of every variable in the data frame college
summary(college) 
#Get descreption for pairs function
?pairs 
#Produce a scatterplot matrix of the first ten columns
pairs(college[,1:10]) 
#Boxplots of Outstate versus Private
plot(college$Outstate ~ college$Private,main = "Outstate vs Private", ylab = "Outstate", xlab = "Private") 
#Create a character vector Elite with a string "No" repeated for number of rows in college data frame
Elite <- rep ("No", nrow(college )) 
#Assign string "Yes" to the Elite vector for each Top10perc variable of college is greater than 50
Elite [college$Top10perc >50] <- "Yes" 
#Encode the Elite vector as a factor
Elite <- as.factor (Elite) 
#Add Elite vector to data frame college. Elite will now appear as a column in college
college <- data.frame(college ,Elite)
#To check how many Elite universities are present
summary(college)
#Plot Outstate vs Elite
plot(college$Outstate ~ college$Elite,main = "Outstate vs Elite", ylab = "Outstate", xlab = "Elite") 
#Divide the print window into 4 regions
par(mfrow=c(2,2)) 
#Histogram for number of pplications variable of college data frame
hist(college$Apps) 
#Histogram for accepted applications variable of college data frame
hist(college$Accept) 
#Histogram for number of enrolled students variable of college data frame
hist(college$Enroll) 
#Histogram for F.Undergrad variable of college data frame 
hist(college$F.Undergrad) 
#Reset par function previously executed
par(mfrow=c(1,1))
#Question 3
#load plyr package
library(plyr) 
#set sf variable to 0 for each of the corresponding year variable lesser than 1954 in baseball data frame
baseball$sf[baseball$year < 1954] = 0 
#Replace "NA" entries of hbp variable by 0
baseball$hbp[is.na(baseball$hbp)] = 0 
#Keep the only rows where ab variable is greater than 50 in baseball data frame
baseball = baseball[baseball$ab > 50,] 
#Add a column obp to baseball data frame and calculate the respective value
baseball$obp=(baseball$h+baseball$bb+baseball$hbp)/(baseball$ab+baseball$bb+baseball$hbp+baseball$sf) 
#Sort the baseball data frame according to obp variable
baseball = baseball[with(baseball, order(obp)),]
#Print top 5 rows with columns id,year and obp from baseball data frame
head(baseball[,c("id","year","obp")],5)
#Question 4
#Load datasets package
library("datasets") 
#Plot Magnitude vs Depth from quake data frame
plot(quakes$mag ~ quakes$depth,xlab="Depth",ylab="Magnitude") 
#Compute the average earthquake depth for each magnitude level
quakeAvgDepth=aggregate(quakes$depth ~ quakes$mag, data = quakes, mean) 
#Rename the first column of quakeAvgDepth to "mag_level"
colnames(quakeAvgDepth)[1] = "mag_level" 
#Rename the second column of quakeAvgDepth to "avg_eq_depth"
colnames(quakeAvgDepth)[2] = "avg_eq_depth" 
#Plot mag_level vs avg_eq_depth of quakeAvgDepth dataframe
plot(quakeAvgDepth$mag_level ~ quakeAvgDepth$avg_eq_depth,xlab="Average Depth",ylab="Magnitude")


#There is a relation between earthquake depth and magnitude. In the seconf graph as the average depth increases magnitude decreses on a overlook