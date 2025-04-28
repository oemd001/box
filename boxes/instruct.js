import { expect, Locator, Page } from '@playwright/test';

export interface DealData {
  region?: string;           …
  dealType?: 'IPO' | 'FO' | 'BT' | 'ABO' | 'CB' | 'CP';
  executionType?: 'Marketed' | 'Bought' | 'Accelerated';
  issuerName?: string;
  issuerNationality?: string;
  issuerIndustrySector?: string;
}

export class Common {
  static async scrollUpToTop(page: Page) {
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'auto' }));
  }

  static async safeClick(el: Locator, timeout = 6_000) {
    await el.waitFor({ state: 'visible', timeout });
    await el.click();
  }

  static async waitForOverlayToDisappear(page: Page) {
    await page.locator('.loading-overlay.loading-full-screen').waitFor({ state: 'detached' });
  }
}

export class CreateDealPage {
  private readonly page: Page;

  // locators 
  private regionSelect          : Locator;
  private dealIcons             : Record<Required<DealData>['dealType'], Locator>;
  private dealDetailsBackButton : Locator;

  //  Deal details locators 
  private issuerNameInput       : Locator;
  private issuerNationalitySelect: Locator;
  private issuerIndustrySelect  : Locator;
  private pricingDatePrefix     : Locator;
  private expectedPricingDate   : Locator;
  private createDealBtn         : Locator;

  constructor(page: Page) {
    this.page = page;

    // selection / navigation
    this.regionSelect  = page.locator('#regionSelect');
    this.dealIcons     = {
      IPO : page.locator('div:has-text("IPO")'),
      FO  : page.locator('div:has-text(/^FOFollow-On$/)'),
      BT  : page.locator('div:has-text("Block Trade")'),
      ABO : page.locator('div:has-text("ABO")'),
      CB  : page.locator('div:has-text("CB")'),
      CP  : page.locator('div:has-text("CP")'),
    };
    this.dealDetailsBackButton = page.locator('#dealDetailsBackButton');

    // deal details (a *very* small subset – extend as needed)
    this.issuerNameInput        = page.locator('#legalNameEntryBox input');
    this.issuerNationalitySelect = page.locator('#issuerNationalitySelect');
    this.issuerIndustrySelect    = page.locator('#issuerIndustrySectorSelect');
    this.pricingDatePrefix       = page.locator('#pricingDatePrefix');
    this.expectedPricingDate     = page.locator('#expectedPricingDate');
    this.createDealBtn           = page.getByRole('button', { name: 'Create Deal' });
  }

  // ---------- high-level actions ----------
  async waitTillPageCompletelyLoaded() {
    await Common.waitForOverlayToDisappear(this.page);
  }

  async selectRegion(region: string = 'Americas') {
    await Common.safeClick(this.regionSelect);
    await this.page.getByRole('option', { name: region }).click();
  }

  async selectDealType(dealType: DealData['dealType'] = 'IPO') {
    await Common.safeClick(this.dealIcons[dealType]);
  }

  async enterDealDetails(dealData: DealData = {}) {
    const {
      issuerName           = `E2E TEST ${Date.now()}`,
      issuerNationality    = 'United States',
      issuerIndustrySector = 'TMT',
    } = dealData;

    // issuer
    await this.issuerNameInput.fill(issuerName);
    await Common.safeClick(this.issuerNationalitySelect);
    await this.page.getByRole('option', { name: issuerNationality }).click();
    await Common.safeClick(this.issuerIndustrySelect);
    await this.page.getByRole('option', { name: issuerIndustrySector }).click();

    // date
    await Common.safeClick(this.pricingDatePrefix);
    await this.page.getByRole('option', { name: 'Week of' }).click();
    await Common.safeClick(this.expectedPricingDate);
    await this.page.getByRole('option', { name: /day-21/ }).click();

    // let the caller decide whether to press the final “Create Deal” button
  }

  async assertCancelOrExitedDeal() {
    await expect(this.issuerNameInput).toHaveValue('');
    await expect(this.pricingDatePrefix).toHaveText(/Select\.\.\./);
  }
}
