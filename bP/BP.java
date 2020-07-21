package bP;

import java.util.Random;

public class BP {
	public static int num_train_sample=60000;
	public static double [][] train_images;
	public static int [] train_labels;
	public static int num_test_sample=10000;
	public static double [][] test_images;
	public static int [] test_labels;
	public static int [] in_label;
	public static int [] answer;//answer[k]表示样本k得到的分类结果
	
	public static int train_time=0;
	
	public static int num_in=28*28;
	public static int num_hidden=30;//隐层结点数
	public static int num_out=10;
	private static double [] in_in;//输入层输入
	private static double [] in_out;//输入层输出
	private static double [] hidden_in;//隐层输入
	private static double [] hidden_out;//隐层输出
	private static double [] out_in;//输出层输入
	private static double [] out_out;//输出层输出
	private static double [][] wt_in_hidden;//输入层到隐层权重
	private static double [][] wt_hidden_out;//隐层到输出层权重
	private static double [] delta_k;
	private static double [][] delta_j;
	private static int num_right;
	public static double ACCURACY;
	private static double STUDY_RATE=0.1;
	private static double WT_IN_HIDDEN=0.01;
	private static double WT_HIDDEN_OUT=0.01;
	
	BP(){	
		in_in=new double[num_in];
		in_out=new double[num_in+1];
		hidden_in=new double[num_hidden];
		hidden_out=new double[num_hidden+1];
		out_in=new double[num_out];
		out_out=new double[num_out];
		in_label=new int[num_out];
		answer=new int[num_train_sample];
		delta_k=new double[num_out];
		delta_j=new double[num_hidden+1][num_out];
	}
//	public static void bp_init(int n) {
//		num_hidden=n;
//		in_in=new double[num_in];
//		in_out=new double[num_in+1];
//		hidden_in=new double[num_hidden];
//		hidden_out=new double[num_hidden+1];
//		out_in=new double[num_out];
//		out_out=new double[num_out];
//		in_label=new int[num_out];
//		answer=new int[num_train_sample];
//		delta_k=new double[num_out];
//		delta_j=new double[num_hidden+1][num_out];
//	}
	
	public static void readin(int k,double [][] images,int [] labels) {
		for(int i=0;i<num_in;i++) {
			in_in[i]=images[k][i];
			in_out[i]=images[k][i];
		}
		in_out[num_in]=1;
		for(int i=0;i<num_out;i++) in_label[i]=0;
		in_label[(int)labels[k]]=1;
	}
	
	public static void init_wt_in_hidden() {
		Random ra=new Random();
		//初始化输入层到隐层权重
		wt_in_hidden=new double[num_in+1][num_hidden];
		for(int i=0;i<num_in+1;i++) {
			for(int j=0;j<num_hidden;j++) {
				wt_in_hidden[i][j]=ra.nextDouble()*WT_IN_HIDDEN-0.5*WT_IN_HIDDEN;
			}
		}
	}
	
	public static void init_wt_hidden_out() {
		Random ra=new Random();
		//初始化隐层到输出层权重
				wt_hidden_out=new double[num_hidden+1][num_out];
				for(int i=0;i<num_hidden+1;i++) {
					for(int j=0;j<num_out;j++) {
						wt_hidden_out[i][j]=ra.nextDouble()*WT_HIDDEN_OUT-0.5*WT_HIDDEN_OUT;
					}
				}
	}
	
	public static double sigmoid(double net) {
		double ans=0;
		ans=1/(1+Math.pow(Math.E, -net));
		return ans;//激励函数
	}
	
	public static void inTohidden() {
		hidden_out[num_hidden]=1;
		for(int i=0;i<num_hidden;i++) {
			hidden_in[i]=0;
			for(int j=0;j<num_in+1;j++) {
				hidden_in[i]+=in_out[j]*wt_in_hidden[j][i];
			}
			hidden_out[i]=sigmoid(hidden_in[i]);
		}
		//System.out.println("intohidden");
	}
	
	public static void hiddenToout() {
		for(int i=0;i<num_out;i++) {
			out_in[i]=0;
			for(int j=0;j<num_hidden+1;j++) {
				out_in[i]+=hidden_out[j]*wt_hidden_out[j][i];
			}
			out_out[i]=sigmoid(out_in[i]);
		}
	}
	
	public static void adjust_hidden_wt() {
		double d_hidden_wt;
		double sigmaj;
		double dJy;
		for(int i=0;i<num_in+1;i++) {
			d_hidden_wt=0;
			for(int j=0;j<num_hidden;j++) {
				dJy=0;
				for(int k=0;k<num_out;k++) {
					dJy+=delta_k[k]*wt_hidden_out[j][k];
				}
				sigmaj=dJy*hidden_out[j]*(1-hidden_out[j]);
				d_hidden_wt=sigmaj*in_out[i];
				wt_in_hidden[i][j]-=d_hidden_wt*STUDY_RATE;                                                                                                                                                             
			}
		}
	}
	
	public static void adjust_out_wt() {
		double d_out_wt;
		for(int k=0;k<num_out;k++) {
			delta_k[k]=-(in_label[k]-out_out[k])*out_out[k]*(1-out_out[k]);
			for(int i=0;i<num_hidden+1;i++) {
				d_out_wt=0;
				d_out_wt=delta_k[k]*hidden_out[i];
				wt_hidden_out[i][k]-=d_out_wt*STUDY_RATE;
			}
		}
	}
	
	public static void getanswer(int k) {
		answer[k]=0;
		double sum=out_out[0];
		for(int i=1;i<num_out;i++) {
			if(out_out[i]>sum) {
				sum=out_out[i];
				answer[k]=i;
			}
		}
	}
	
	public static double right(int[] labels,int num) {
		train_time++;
		double num_right=0;
		for(int i=0;i<num;i++) {
			if(answer[i]==labels[i]) num_right++;
		}
		double accu=num_right/num;
		//System.out.printf("当前训练轮次：%d,正确率：%f\n", train_time,accu);
		return accu;
	}//计算当前分类器的正确率
	
	public static double train() {
//		Random rand=new Random();
//		int k=-1;
//		while(true) {
//			for(int i=0;i<num_train_sample;i++) {
//				readin(i,train_images,train_labels);
//				inTohidden();
//				hiddenToout();
//				getanswer(i);
//			}
//			if(right(train_labels,num_train_sample)<ACCURACY) {
//				k=rand.nextInt(num_train_sample);
//				//k=(k+1)%num_train_sample;
//				readin(k,train_images,train_labels);
//				inTohidden();
//				hiddenToout();
//				adjust_out_wt();
//				adjust_hidden_wt();
//			}
//			else break;
//		}
		int turn=0;
		double accu;
		do {
			turn++;
			for(int i=0;i<num_train_sample;i++) {
				readin(i,train_images,train_labels);
				inTohidden();
				hiddenToout();
				getanswer(i);
				adjust_out_wt();
				adjust_hidden_wt();
			}
			accu=right(train_labels,num_train_sample);
			System.out.printf("第%d轮学习结束，正确率为：%f\n",turn,accu);
		}while(accu<ACCURACY);
		return accu;
	}
	
	public static double test() {
		for(int i=0;i<num_test_sample;i++) {
			readin(i,test_images,test_labels);
			inTohidden();
			hiddenToout();
			getanswer(i);
		}
		double accu=right(test_labels,num_test_sample);
		System.out.println("测试集上的精度为："+accu);
		return accu;
	}
	
}
