using CSV
using JuMP
using CPLEX
using DataFrames
using Random

include("functions.jl")

dataSet = ARGS[1]
dataFolder = "./data/"
resultsFolder = "./res/"

# Create the features tables (or load them if they already exist)
# Note: each line corresponds to an individual, the 1st column of each table contain the class
# Details:
# - read the file ./data/kidney.csv
# - save the features in ./data/kidney_test.csv and ./data/kidney_train.csv
createFeaturesTime = time()
train, test = createFeatures(dataFolder, dataSet)
createFeaturesTime = time() - createFeaturesTime

# Create the rules (or load them if they already exist)
# Note: each line corresponds to a rule, the first column corresponds to the class
# Details:
# - read the file ./data/kidney_train.csv
# - save the rules in ./res/kidney_rules.csv
createRulesTime = time()
rules = createRules(dataSet, resultsFolder, train)
createRulesTime = time() - createRulesTime

# Order the rules (limit the resolution to 300 seconds)
# Details:
# - read the file ./data/kidney_rules.csv
# - save the rules in ./res/kidney_ordered_rules.csv
timeLimitInSeconds = 300

sortRulesTime = time()
orderedRules = sortRules(dataSet, resultsFolder, train, rules, timeLimitInSeconds)
sortRulesTime = time() - sortRulesTime

println("-- Computing time")
println("createFeaturesTime:\t", round(createFeaturesTime, digits=2), " seconds")
println("createRulesTime:\t", round(createRulesTime, digits=2), " seconds")
println("sortRulesTime:\t\t", round(sortRulesTime, digits=2), " seconds\n")

println("-- Train results")
showStatistics(orderedRules, train)

println("-- Test results")
showStatistics(orderedRules, test)
