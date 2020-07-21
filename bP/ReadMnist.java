package bP;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import javax.imageio.ImageIO;

import java.io.FileWriter;
import java.io.BufferedWriter;


public class ReadMnist {

    public static final String TRAIN_IMAGES_FILE = "D:/mnist/train-images.idx3-ubyte";
    public static final String TRAIN_LABELS_FILE = "D:/mnist/train-labels.idx1-ubyte";
    public static final String TEST_IMAGES_FILE = "D:/mnist/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_FILE = "D:/mnist/t10k-labels.idx1-ubyte";

    /**
     * change bytes into a hex string.
     *
     * @param bytes bytes
     * @return the returned hex string
     */
    public static String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }

    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static double[][] getImages(String fileName) {
        double[][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new double[number][xPixel * yPixel];
                for (int i = 0; i < number; i++) {
                    double[] element = new double[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {
                        //element[j] = bin.read();                                // 逐一读取像素值
                        // normalization
                        element[j] = bin.read() / 255.0;
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }

    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return
     */
    public static int[] getLabels(String fileName) {
        int[] y = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                y = new int[number];
                for (int i = 0; i < number; i++) {
                    y[i] = bin.read();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }
    
    /**
     * draw a gray picture and the image format is JPEG.
     *
     * @param pixelValues pixelValues and ordered by column.
     * @param width       width
     * @param high        high
     * @param fileName    image saved file.
     */
    public static void drawGrayPicture(int[] pixelValues, int width, int high, String fileName) {
        BufferedImage bufferedImage = new BufferedImage(width, high, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < high; j++) {
                int pixel = 255 - pixelValues[i * high + j];
                int value = pixel + (pixel << 8) + (pixel << 16);   // r = g = b 时，正好为灰度
                bufferedImage.setRGB(j, i, value);
            }
        }
        try{
        	ImageIO.write(bufferedImage, "JPEG", new File(fileName));
        }
        catch(IOException e) {
        	System.out.println("IOException!");
        }
    }


    public static void main(String[] args) {
    	BP bp=new BP();
        BP.train_images = getImages(TRAIN_IMAGES_FILE);
        BP.train_labels = getLabels(TRAIN_LABELS_FILE);

        BP.test_images = getImages(TEST_IMAGES_FILE);
        BP.test_labels = getLabels(TEST_LABELS_FILE);

        try {
        	String ACCU,train_accu,test_accu;
            File file = new File("javaio-appendfile.txt");
            //if file doesnt exists, then create it
            if(!file.exists()){
             file.createNewFile();
            }
            //true = append file
            FileWriter fileWritter = new FileWriter(file.getName(),true);

//          drawGrayPicture(train_images[10], 28, 28, "D:/mnistpic");
            BP.ACCURACY=0.97;
//    		for(int i=0;i<30;i++) {
//    			BP.bp_init(10+i);
//    			ACCU=String.valueOf(BP.num_hidden);
//    			fileWritter.write(ACCU+"\t");
//    			System.out.println("num_hidden:"+BP.num_hidden);
    			BP.init_wt_in_hidden();
    			BP.init_wt_hidden_out();
    			train_accu=String.valueOf(BP.train());
    			fileWritter.write(train_accu+"\t");
    			test_accu=String.valueOf(BP.test());
    			fileWritter.write(test_accu+"\n");
//    		}
            fileWritter.close();
        }
        catch(IOException e) {
        	e.printStackTrace();
        }
        System.out.println("Done");
    }
}
