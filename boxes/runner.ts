import { test, expect } from '@playwright/test';
import { CreateDealPage, Common } from '../pages/createDeal.page';

test.describe('Deal Creation:', () => {

  test('Should be able to navigate back to deal-type selection during create', async ({ page }) => {

    // arrange
    const createDealPage = new CreateDealPage(page);
    await page.goto('https://qa.ibdweb.site.gs.com/ecm-syndicate-link/#/deal/dealSetup/');
    await createDealPage.waitTillPageCompletelyLoaded();
    await createDealPage.selectRegion('Americas');
    await createDealPage.selectDealType('FO');

    // act
    await createDealPage.enterDealDetails();
    await Common.scrollUpToTop(page);
    await Common.safeClick(createDealPage['dealDetailsBackButton']);

    // assert
    await createDealPage.selectRegion();      
    await createDealPage.selectDealType();
    await createDealPage.assertCancelOrExitedDeal();
  });

});
