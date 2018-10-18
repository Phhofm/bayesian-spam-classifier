import java.io.File;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;

import static java.lang.Boolean.TRUE;

/**
 * Created by IntelliJ IDEA.
 * User: verman
 * Date: 5/4/2014
 * Time: 10:41 AM
 * To change this template use File | Settings | File Templates.
 */
public class BayesianClassifier {

    /**
     * This is a spam-filter exercise of the lecture 'Practical Artificial Intelligence' at UZH.
     * I chose the Binarized Multinomial Naive Bayes, because according to the paper (Spam Filtering with Naive Bayes – Which Naive Bayes?) this approach gave the best overall results compared to all others
     * For training data, the Enron2 dataset is used (http://www2.aueb.gr/users/ion/data/enron-spam/) like in the paper.
     *
     * This program uses first the trainingdata. It extracts all words from all messages and then builds a vector of the most used words (# of words used is declared by the user, see the "NUMBEROFTOKENS" variable.
     * Afterwards it builds a binarized vector for every single message in the training data, using the vector we built with the most used words. So it evaluates in every message if a token appears or not
     * It then uses the Binarized Multinomial Naive Bayes to classify each message in the testing folders. See the in-code comment for more details in what happens.
     * I chose the Binarized Multinomial Naive Bayes, because according to the paper (Spam Filtering with Naive Bayes – Which Naive Bayes?) this approach gave the best overall results compared to all others
     * author: Philip Hofmann, 14-710-842
     */

    //defined variables I will use
    private final int NUMBEROFTOKENS = 450;                             //this is the number of tokens that will be considered for training and testing. This number will greatly influence the training (/processing) time. The bigger the longer this algo takes.
    private final double THRESHHOLD = 0.6;                              //the threshhold used for evaluating if a message is spam
    private ArrayList<String> allWords = new ArrayList<>();             //allWords it can find in all emails. The most often used words will be used for our binary vector for the binarized multinomial naive bayes
    private ArrayList<String> vectorWords = new ArrayList<>();          //the words we actually finally use for the binary vectors
    private ArrayList<List<Boolean>> spamVectors = new ArrayList<>();   //this will hols all the binary vectors of the spam category. each message will be one vector
    private ArrayList<List<Boolean>> hamVectors = new ArrayList<>();    //this will hols all the binary vectors of the ham category. each message will be one vector
    private int numberOfSpam;                                           //the total number of spam messages in training set; used for prior probability
    private int numberOfHam;                                            //the total number of ham messages; used for prior probability
    private int numberOfTotalMessages;                                  //the total number of messages in training set. used for classifier
    private double priorSpam;                                           //the prior probability that a new message is spam
    private double priorHam;                                            //the prior probability that a new message is ham
    private List<Integer> tokenOccurenceSpam = new ArrayList<>();       //for Nt,c in classifier algo. Number of occurences of token in class Spam
    private List<Integer> tokenOccurenceHam = new ArrayList<>();        //Nt,c in classifier for class Ham
    private int totalTokenOccurenceSpam = 0;                            //Nc for class Spam
    private int totalTokenOccurenceHam = 0;                             //Nc for class Ham


    public File[] filterFiles(File[] initialFiles) {

        Vector listOfFiles = new Vector();
        for (int i = 0; i < initialFiles.length; i++) {
            if (initialFiles[i].getName().endsWith(".txt")) {
                listOfFiles.addElement(initialFiles[i]);
            }
        }

        File[] fileArray = new File[listOfFiles.size()];
        listOfFiles.toArray(fileArray);
        return fileArray;

    }

    public void train(String spamTrainingFolder, String hamTrainingFolder) {

        File spamTrainingDirectory = new File(spamTrainingFolder);
        if (!spamTrainingDirectory.exists()) {
            System.out.println("ERR: The Spam Training Directory does not exist");
            return;
        }

        File hamTrainingDirectory = new File(hamTrainingFolder);
        if (!hamTrainingDirectory.exists()) {
            System.out.println("ERR: The Ham Training Directory does not exist");
            return;
        }

        File spamFiles[] = filterFiles(spamTrainingDirectory.listFiles());

        int numberOfFiles = 0;

        //get the number of files in Spam folder and add all the words to the allwords arraylist
        numberOfFiles = getAllWordsInFilesAndNumberOfFiles(spamFiles, numberOfFiles, "SpamTrainingFolder\\");
        System.out.println(numberOfFiles + " files found in spam training folder");
        numberOfSpam = numberOfFiles;

        //just to check
        System.out.println(allWords.size() + " words in the allwords arraylist");


        numberOfFiles = 0;
        File hamFiles[] = filterFiles(hamTrainingDirectory.listFiles());
        //get the number of files in Ham folder and add all the words to the allwords arraylist
        numberOfFiles = getAllWordsInFilesAndNumberOfFiles(hamFiles, numberOfFiles, "HamTrainingFolder\\");
        System.out.println(numberOfFiles + " files found in ham training folder");
        numberOfHam = numberOfFiles;

        //just to check
        System.out.println(allWords.size() + " final words in the allwords arraylist");

        //find the most used words in the list because they will be most significant for our vector
        Map<String, Long> map = allWords.stream().collect(Collectors.groupingBy(w -> w, Collectors.counting()));
        List<Map.Entry<String, Long>> mostUsedWords = map.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).limit(NUMBEROFTOKENS).collect(Collectors.toList());
        System.out.println(NUMBEROFTOKENS + " tokens used for evaluating each message");

        //read the most used words into our arraylist vectorlist. we will use them to classify
        for (Map.Entry<String, Long> mostUsedWord : mostUsedWords) {
            vectorWords.add(mostUsedWord.getKey());
        }

        //we loop again through all the spam messages because our vector is now defined. We generate from each message a boolean vector and add it to the vector list
        vectorizeMessages(spamFiles, spamFiles[0], "SpamTrainingFolder\\", spamVectors);
        System.out.println(spamVectors.size() + " vectors in the SpamArrayList");

        //we loop again through all the ham messages because our vector is now defined. We generate from each message a boolean vector and add it to the vector list
        vectorizeMessages(hamFiles, spamFiles[0], "HamTrainingFolder\\", hamVectors);
        System.out.println(hamVectors.size() + " vectors in the HamArrayList");

        //count occurence of each token in each class. Used for classifier later (Nt,c). Remember these are binarized vectore, so it counts the token per message at max 1. Highest value is therefore number of messages in this class.
        for (int token = 0; token < vectorWords.size(); token++) {
            int tokenOccurence = 0;
            for (List<Boolean> spamVector : spamVectors) {
                if (spamVector.get(token).equals(TRUE)) {
                    tokenOccurence++;
                }
            }
            tokenOccurenceSpam.add(tokenOccurence);
        }

        System.out.println("Token Occurence Spam: " + tokenOccurenceSpam);

        for (int token = 0; token < vectorWords.size(); token++) {
            int tokenOccurence = 0;
            for (List<Boolean> hamVector : hamVectors) {
                if (hamVector.get(token).equals(TRUE)) {
                    tokenOccurence++;
                }
            }
            tokenOccurenceHam.add(tokenOccurence);
        }

        System.out.println("Token Occurence Ham: " + tokenOccurenceHam);

        //prior probability for classifier
        numberOfTotalMessages = numberOfSpam + numberOfHam;
        System.out.println("number of total Messages: " + numberOfTotalMessages);
        System.out.println("numberOfSpam: " + numberOfSpam);
        System.out.println("numberOfHam: " + numberOfHam);
        priorSpam = (double) numberOfSpam / numberOfTotalMessages; //we need to convert the inputs to double for the division. only converting one explicitly is enough since the other one will be converted implicitly
        System.out.println("prior Spam: " + priorSpam);
        priorHam = (double) numberOfHam / numberOfTotalMessages; //we need to convert the inputs to double for the division. only converting one explicitly is enough since the other one will be converted implicitly
        System.out.println("prior Ham: " + priorHam);

        //calculate Nc, so the # of all token-appearance for a class c. This is Nc and used for p(t|c).

        for (Integer aTokenOccurenceSpam : tokenOccurenceSpam) {
            totalTokenOccurenceSpam += aTokenOccurenceSpam;
        }
        System.out.println("total # of token occurences for class spam: " + totalTokenOccurenceSpam);


        for (int i = 0; i < tokenOccurenceHam.size(); i++) {
            totalTokenOccurenceHam += tokenOccurenceHam.get(i);
        }
        System.out.println("total # of token occurences for class ham: " + totalTokenOccurenceHam);


    }

    private void vectorizeMessages(File[] spamFiles, File spamFile, String s, ArrayList<List<Boolean>> spamVectors) {
        for (File f : spamFiles) {

            //generate a vector array. this will be used for each message
            List<Boolean> vector = new ArrayList<Boolean>(Arrays.asList(new Boolean[vectorWords.size()]));
            Collections.fill(vector, Boolean.FALSE);

            // read all the words that appear in messages of this class into an arraylist
            File file = new File(spamFile.getAbsolutePath());
            String inputFileName = s + f.getName();
            FileReader reader = null;
            try {
                reader = new FileReader(inputFileName);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            Scanner wordFromFileTest = null;
            if (reader != null) {
                wordFromFileTest = new Scanner(reader);
            }
            if (wordFromFileTest != null) {

                while (wordFromFileTest.hasNext()) {
                    String word = wordFromFileTest.next();
                    //for every single word in the text file, we match it to the vector words and if it matches we set the value of the boolean array position to true
                    for (int i = 0; i < vector.size(); i++) {
                        if (word.equals(vectorWords.get(i))) {
                            vector.set(i, TRUE);
                        }
                    }
                }
            }
            if (wordFromFileTest != null) {
                wordFromFileTest.close();
            }
            //we now add the finalized vector for this message to the vector list
            spamVectors.add(vector);
        }
    }

    private int getAllWordsInFilesAndNumberOfFiles(File[] spamFiles, int numberOfFiles, String s) {
        for (File f : spamFiles) {

            // read all the words that appear in messages of this class into an arraylist
            //we are gonna use this later on to create out vector with most used words/tokens
            //the words of the messages in the ham folder will be added too to this arraylist later on
            String inputFileName = s + f.getName();
            FileReader reader = null;
            try {
                reader = new FileReader(inputFileName);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            Scanner wordFromFileTest = null;
            if (reader != null) {
                wordFromFileTest = new Scanner(reader);
            }
            if (wordFromFileTest != null) {
                while (wordFromFileTest.hasNext()) {
                    allWords.add(wordFromFileTest.next());
                }
            }
            if (wordFromFileTest != null) {
                wordFromFileTest.close();
            }

            numberOfFiles++;

        }
        return numberOfFiles;
    }

    public void test(String spamTestingFolder, String hamTestingFolder) {

        File spamTestingDirectory = new File(spamTestingFolder);
        if (!spamTestingDirectory.exists()) {
            System.out.println("ERR: The Spam Testing Directory does not exist");
            return;
        }

        File hamTestingDirectory = new File(hamTestingFolder);
        if (!hamTestingDirectory.exists()) {
            System.out.println("ERR: The Ham Testing Directory does not exist");
            return;
        }

        System.out.println("Testing phase:");

        int allSpam = 0;
        int SpamClassifiedAsHam = 0; //Spams incorrectly classified as Hams

        File spamFiles[] = filterFiles(spamTestingDirectory.listFiles());
        for (File f : spamFiles) {
            allSpam++;
            if (!isSpam(f))
                SpamClassifiedAsHam++;

        }

        int allHam = 0;
        int HamClassifiedAsSpam = 0; //Hams incorrectly classified as Spams

        File hamFiles[] = filterFiles(hamTestingDirectory.listFiles());
        for (File f : hamFiles) {
            allHam++;
            if (isSpam(f))
                HamClassifiedAsSpam++;

        }

        System.out.println("###_DO_NOT_USE_THIS_###Spam = " + allSpam);
        System.out.println("###_DO_NOT_USE_THIS_###Ham = " + allHam);
        System.out.println("###_DO_NOT_USE_THIS_###SpamClassifAsHam = " + SpamClassifiedAsHam);
        System.out.println("###_DO_NOT_USE_THIS_###HamClassifAsSpam = " + HamClassifiedAsSpam);


    }


    public boolean isSpam(File f) {


        //first, convert message into vector
        //generate a vector array. this will be used for each message
        List<Boolean> vector = new ArrayList<Boolean>(Arrays.asList(new Boolean[vectorWords.size()]));
        Collections.fill(vector, Boolean.FALSE);

        // read all the words that appear in messages of this class into an arraylist
        File file = new File(f.getAbsolutePath());
        String inputFileName;

        if (f.getAbsolutePath().toLowerCase().contains("SpamTestingFolder".toLowerCase())) {
            inputFileName = "SpamTestingFolder\\" + f.getName();
        } else {
            inputFileName = "HamTestingFolder\\" + f.getName();
        }
        FileReader reader = null;
        try {
            reader = new FileReader(inputFileName);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        Scanner wordFromFileTest = null;
        if (reader != null) {
            wordFromFileTest = new Scanner(reader);
        }
        if (wordFromFileTest != null) {
            while (wordFromFileTest.hasNext()) {
                String word = wordFromFileTest.next();
                //for every single word in the text file, we match it to the vector words and if it matches we set the value of the boolean array position to true
                for (int i = 0; i < vector.size(); i++) {
                    if (word.equals(vectorWords.get(i))) {
                        vector.set(i, TRUE);
                    }
                }
            }
        }
        if (wordFromFileTest != null) {
            wordFromFileTest.close();
        }


        //Binarized Multinomial Naive Bayes Classifier

        //variables declaration
        int exponent;
        double dividend;
        double laplaceanPrior;
        double multipleProbability = 0.0;
        double totalProbability;
        double divisor;
        double laplaceanPriorHam;
        double multipleProbabilityHam = 0.0;

        //calculate dividend, see formula in paper
        for (int i = 0; i < vectorWords.size(); i++) {
            exponent = 0;
            laplaceanPrior = (1 + (double) tokenOccurenceSpam.get(i)) / (vectorWords.size() + (double) totalTokenOccurenceSpam);
            if (vector.get(i).equals(TRUE)) {
                exponent = 1;
            }
            laplaceanPrior = Math.pow(laplaceanPrior, exponent);
            if (i == 0) {
                multipleProbability = laplaceanPrior;
            } else {
                multipleProbability *= laplaceanPrior;  //at one point this reaches 0.0 since we multiplicate the whole time with small values. But i thought i implementet the algorithm of the paper pretty well
            }
        }
        dividend = priorSpam * multipleProbability;

        //calculate divisor; see formula in paper
        for (int i = 0; i < vectorWords.size(); i++) {
            exponent = 0;
            laplaceanPriorHam = (1 + (double) tokenOccurenceHam.get(i)) / (vectorWords.size() + (double) totalTokenOccurenceHam);
            if (vector.get(i).equals(TRUE)) {
                exponent = 1;
            }
            laplaceanPriorHam = Math.pow(laplaceanPriorHam, exponent);
            if (i == 0) {
                multipleProbabilityHam = laplaceanPriorHam;
            } else {
                multipleProbabilityHam *= laplaceanPriorHam;  //at one point this reaches 0.0 since we multiplicate the whole time with small values. But i thought i implementet the algorithm of the paper pretty well
            }
        }

        divisor = dividend + multipleProbabilityHam;

        //calculate total probability for message. prevent dividing by zero.
        if (divisor != 0) {
            totalProbability = dividend / divisor;
        } else {
            totalProbability = 0.0;
        }

        //compare to threshhold
        return totalProbability > THRESHHOLD;
    }

}