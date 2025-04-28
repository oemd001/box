// tests/createDeal.e2e.spec.ts
import { test, expect } from '@playwright/test';
import { CreateDealPage, Common, DealData } from '../pages/createDeal.page';

test.describe('Deal Creation – full flow', () => {

  test('should create a deal end-to-end', async ({ page }) => {

    /* ---------- arrange ---------- */
    const createDealPage = new CreateDealPage(page);
    const dealData: DealData = {
      region              : 'Americas',
      dealType            : 'FO',
      issuerName          : `E2E-TEST-${Date.now()}`,
      issuerNationality   : 'United States',
      issuerIndustrySector: 'TMT',
      executionType       : 'Marketed',
    };

    await page.goto(
      'https://qa.ibdweb.site.gs.com/ecm-syndicate-link/#/deal/dealSetup/'
    );
    await createDealPage.waitTillPageCompletelyLoaded();

    /* ---------- deal-setup screen ---------- */
    await createDealPage.selectRegion(dealData.region);
    await createDealPage.selectDealType(dealData.dealType);
    await createDealPage.enterDealDetails(dealData);

    // execution type (Americas + BT need this in prod, but keeping example generic)
    await page.locator('#executionTypeSelect').click();
    await page.getByRole('option', { name: dealData.executionType! }).click();

    await page.getByRole('button', { name: 'Create Deal' }).click();

    /* ---------- pre-pricing tab ---------- */
    await Common.waitForOverlayToDisappear(page);

    // A quick “did we land?” sanity check: deal-name field now has placeholder = our name
    await expect(
      page.locator('#dealNameEntryBox input')
    ).toHaveAttribute('placeholder', dealData.issuerName);

    /* ---------- extra assertions / cleanup ---------- */
    // If your app shows a toast or navigates to a URL with the dealId,
    // add those expectations here.  Example:
    //
    // const url = page.url();
    // expect(url).toMatch(/\/deal\/\d+\/prePricing/);
  });

});
