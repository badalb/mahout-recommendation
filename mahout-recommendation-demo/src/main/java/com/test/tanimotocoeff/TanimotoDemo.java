package com.test.tanimotocoeff;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.List;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.BooleanPreference;
import org.apache.mahout.cf.taste.impl.model.BooleanUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class TanimotoDemo {

	private static long CUSTOMER_A = 0;
	private static long CUSTOMER_B = 1;
	private static long CUSTOMER_C = 2;
	private static long CUSTOMER_D = 3;
	private static long CUSTOMER_E = 4;

	private static long productOne = 0;
	private static long productTwo = 1;
	private static long productThree = 2;
	private static long productFour = 3;
	private static long productFive = 4;

	public static FastByIDMap<PreferenceArray> setup() {
		FastByIDMap<PreferenceArray> userIdMap = new FastByIDMap<PreferenceArray>();

		BooleanUserPreferenceArray customerAPrefs = new BooleanUserPreferenceArray(
				4);
		customerAPrefs.set(0, new BooleanPreference(CUSTOMER_A, productOne));
		customerAPrefs.set(1, new BooleanPreference(CUSTOMER_A, productTwo));
		customerAPrefs.set(2, new BooleanPreference(CUSTOMER_A, productFour));
		customerAPrefs.set(3, new BooleanPreference(CUSTOMER_A, productFive));

		BooleanUserPreferenceArray customerBPrefs = new BooleanUserPreferenceArray(
				3);
		customerBPrefs.set(0, new BooleanPreference(CUSTOMER_B, productTwo));
		customerBPrefs.set(1, new BooleanPreference(CUSTOMER_B, productThree));
		customerBPrefs.set(2, new BooleanPreference(CUSTOMER_B, productFive));

		BooleanUserPreferenceArray customerCPrefs = new BooleanUserPreferenceArray(
				2);
		customerCPrefs.set(0, new BooleanPreference(CUSTOMER_C, productOne));
		customerCPrefs.set(1, new BooleanPreference(CUSTOMER_C, productFive));

		BooleanUserPreferenceArray customerDPrefs = new BooleanUserPreferenceArray(
				3);
		customerDPrefs.set(0, new BooleanPreference(CUSTOMER_D, productOne));
		customerDPrefs.set(1, new BooleanPreference(CUSTOMER_D, productThree));
		customerDPrefs.set(2, new BooleanPreference(CUSTOMER_D, productFive));

		BooleanUserPreferenceArray customerEPrefs = new BooleanUserPreferenceArray(
				3);
		customerEPrefs.set(0, new BooleanPreference(CUSTOMER_E, productOne));
		customerEPrefs.set(1, new BooleanPreference(CUSTOMER_E, productThree));
		customerEPrefs.set(2, new BooleanPreference(CUSTOMER_E, productFive));

		userIdMap.put(CUSTOMER_A, customerAPrefs);
		userIdMap.put(CUSTOMER_B, customerBPrefs);
		userIdMap.put(CUSTOMER_C, customerCPrefs);
		userIdMap.put(CUSTOMER_D, customerDPrefs);
		userIdMap.put(CUSTOMER_E, customerEPrefs);

		return userIdMap;
	}

	public static void testSimilarities(ItemSimilarity tanimoto)
			throws Exception {
		assertEquals((double) 1,
				tanimoto.itemSimilarity(productOne, productOne), 0.01);
		assertEquals((double) 1 / 3,
				tanimoto.itemSimilarity(productOne, productTwo), 0.01);
		assertEquals((double) 0,
				tanimoto.itemSimilarity(productOne, productThree), 0.01);
		assertEquals((double) 1 / 2,
				tanimoto.itemSimilarity(productOne, productFour), 0.01);
		assertEquals((double) 2 / 3,
				tanimoto.itemSimilarity(productOne, productFive), 0.01);

		assertEquals((double) 1 / 1,
				tanimoto.itemSimilarity(productTwo, productTwo), 0.01);
		assertEquals((double) 1 / 2,
				tanimoto.itemSimilarity(productTwo, productThree), 0.01);
		assertEquals((double) 1 / 2,
				tanimoto.itemSimilarity(productTwo, productFour), 0.01);
		assertEquals((double) 2 / 3,
				tanimoto.itemSimilarity(productTwo, productFive), 0.01);

		assertEquals((double) 1,
				tanimoto.itemSimilarity(productThree, productThree), 0.01);
		assertEquals((double) 0,
				tanimoto.itemSimilarity(productThree, productFour), 0.01);
		assertEquals((double) 1 / 3,
				tanimoto.itemSimilarity(productThree, productFive), 0.01);

		assertEquals((double) 1,
				tanimoto.itemSimilarity(productFour, productFour), 0.01);
		assertEquals((double) 1 / 3,
				tanimoto.itemSimilarity(productFour, productFive), 0.01);

		assertEquals((double) 1,
				tanimoto.itemSimilarity(productFive, productFive), 0.01);
	}

	public static void testRecommendProducts(DataModel dataModel,
			ItemSimilarity tanimoto) throws Exception {

		ItemBasedRecommender recommender = new GenericItemBasedRecommender(
				dataModel, tanimoto);

		List<RecommendedItem> similarToProductThree = recommender
				.mostSimilarItems(productThree, 4);

		System.out.println("##### Item similarity for product three(2) ######");

		for (RecommendedItem item : similarToProductThree) {
			System.out.println(item.getItemID());
		}
	}

	public static void testUserRecommendProducts() throws Exception {

		DataModel dataModel = new FileDataModel(
				new File(
						"/Users/badalb/projects/Embrace/mahout-demo/src/main/resources/dataset.csv"));

		UserSimilarity similarity = new TanimotoCoefficientSimilarity(dataModel);
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1,
				similarity, dataModel);
		UserBasedRecommender recommender = new GenericUserBasedRecommender(
				dataModel, neighborhood, similarity);
		List<RecommendedItem> recommendations = recommender.recommend(5, 3);

		System.out.println("##### Item recommendation for user(1) ######");

		for (RecommendedItem recommendation : recommendations) {
			System.out.println(recommendation.getItemID());
		}

	}

	public static void main(String[] args) {
		try {
			DataModel dataModel = new GenericDataModel(setup());
			ItemSimilarity tanimoto = new TanimotoCoefficientSimilarity(
					dataModel);
			;

			setup();
			testRecommendProducts(dataModel, tanimoto);
			testUserRecommendProducts();

		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}
}